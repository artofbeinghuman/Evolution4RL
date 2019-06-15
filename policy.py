import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import json
from tensorboardX import SummaryWriter
import datetime


class SimpleLogger(object):
    def __init__(self, writer):
        self.writer = writer

    def __call__(self, text):
        writer.add_text('[LOG]', text)


date = datetime.datetime.now()
writer = SummaryWriter("runs/ESAtari-", comment=date)
log = SimpleLogger(writer)


class SharedNoiseTable(object):
    def __init__(self, seed):
        import ctypes
        import multiprocessing
        # seed = 123
        count = 250000000  # 1 gigabyte of 32-bit numbers. Will actually sample 2 gigabytes below.
        log('Sampling {} random numbers with seed {}'.format(count, seed))
        self._shared_mem = multiprocessing.Array(ctypes.c_float, count)
        self.noise = np.ctypeslib.as_array(self._shared_mem.get_obj())
        assert self.noise.dtype == np.float32
        self.noise[:] = np.random.RandomState(seed).randn(count)  # 64-bit to 32-bit conversion here
        log('Sampled {} bytes'.format(self.noise.size * 4))

    def get(self, i, dim):
        return self.noise[i:i + dim]

    def sample_index(self, stream, dim):
        return stream.randint(0, len(self.noise) - dim + 1)


# for virtual batch normalization, we need to sample a batch of (random) observations over which we calculate the parameters
# for batch normalization, which will then be held fixed when appling batchnorm to the actual policy evaluation/agent run.
def get_ref_batch(env, batch_size=128, p=0.05):
    """
    Performs random actions in <env> and adds the subsequent observations with
    a probability of <p> to the reference batch, which is going to be output
    once it is <batch_size> long.
    """
    ref_batch = []
    ob = env.reset()
    while len(ref_batch) < batch_size:
        ob, _, done, _ = env.step(env.action_space.sample())
        if done:
            ob = env.reset()
        elif np.random.RandomState(0).rand() < p:
            ref_batch.append(ob)
    return torch.tensor(np.asarray(ref_batch).transpose(0, 3, 1, 2))


def to_obs_tensor(single_observation):
    return torch.tensor(single_observation.transpose(2, 0, 1)[np.newaxis, :, :, :])


def run(exp_file):
    with open(exp_file, 'r') as f:
        exp = json.loads(f.read())

    return exp


class VirtualBatchNorm(nn.Module):
    """
    Module for Virtual Batch Normalization over batch dimension, Input: batch_size x [feature_dims],
    without affine transformation (no learnt gamma, beta parameters in this rl setting)
    adapted from https://github.com/deNsuh/segan-pytorch/blob/master/vbnorm.py
    """

    def __init__(self):
        super(VirtualBatchNorm, self).__init__()
        # batch statistics
        self.eps = 1e-5  # epsilon
        self.ref_mean = None
        self.ref_var = None

        self.update = True

    def get_stats(self, x):
        mean = x.mean(0, keepdim=True)
        var = (x ** 2).mean(0, keepdim=True) - mean**2
        return mean, var

    def forward(self, x):
        if self.update:
            mean, var = self.get_stats(x)
            if self.ref_mean is not None or self.ref_var is not None:
                batch_size = x.shape[0]
                coeff = 1. / (batch_size + 1.)
                self.ref_mean = coeff * mean + (1 - coeff) * self.ref_mean
                self.ref_var = coeff * var + (1 - coeff) * self.ref_var
            else:
                self.ref_mean, self.ref_var = mean, var
            return self._normalize(x, self.ref_mean, self.ref_var)
        else:
            return self._normalize(x, self.ref_mean, self.ref_var)

    def _normalize(self, x, mean, var):
        return x - mean / torch.sqrt(var + self.eps)

    def __repr__(self):
        return ('{}()'.format(self.__class__.__name__))


class Policy(nn.Module):
    def __init__(self, env):
        super(Policy, self).__init__()

        self.env = env
        self._ref_batch = None
        input_shape = self.env.observation_space.shape
        output_shape = self.env.action_space.n
        self.conv = nn.Sequential(nn.Conv2d(in_channels=input_shape[-1], out_channels=16,
                                            kernel_size=8, stride=4),
                                  #nn.BatchNorm2d(num_features=16, track_running_stats=False),
                                  VirtualBatchNorm(),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=16, out_channels=32,
                                            kernel_size=4, stride=2),
                                  #nn.BatchNorm2d(num_features=32, track_running_stats=False),
                                  VirtualBatchNorm(),
                                  nn.ReLU())

        def conv_output(width, kernel, stride, padding=0):
            return int((width - kernel + 2 * padding) // stride + 1)

        self.lin_dim = (conv_output(conv_output(input_shape[-2], 8, 4), 4, 2))**2 * 32

        self.mlp = nn.Sequential(nn.Linear(in_features=self.lin_dim, out_features=256),
                                 # nn.BatchNorm1d(num_features=256),
                                 VirtualBatchNorm(),
                                 nn.ReLU(),
                                 nn.Linear(in_features=256, out_features=output_shape))

        self.apply(self.initialise_parameters)

    def forward(self, obs):
        # expect an input of form: N x C x H x W
        # where N should be 1 (except for ref_batch), H=W=84 as per atari_wrappers.wrap_deepmind() and C=4 due to framestacking of 4 greyscale frames
        x = self.conv(obs)
        x = x.view(-1, self.lin_dim)
        x = self.mlp(x)
        return torch.argmax(x, dim=1)  # action not chosen stochastically

    def rollout(self):
        """
        Eventuell rollout in eine andere Klasse packen. Ist Rollout wirklich ein teil des torch.module Policy? Evtl in ES

        sich überlegen, einen self.ref_batch zu haben, der am anfang leer ist, bei jedem rollout wird dann zu erst der aktuelle
        ref_batch durchgegeben, um VBN zu aktualisieren, bevor VBN dann gefreezt wird und der eigentliche rollout beginnt.
        Beim ersten rollout muss der ref_batch erst noch gesampled werden, wie es aktuell in initialise_policy passiert. Nachdem
        VBN mit dem aktuellen ref_batch aktualisiert wurde, wird er wieder geleert.
        Während des rollouts, wird dann mit p=0.001 wahrscheinlichkeit jede observation in self.ref_batch gesammelt, sodass bei
        einem nächsten rollout VBN mit diesem ref_batch aktualisiert werden kann.
        """
        # initialise virtual batch normalization
        self.freeze_VBN(False)
        if self._ref_batch is None:
            self._ref_batch = get_ref_batch(self.env)
        self.forward(self._ref_batch)
        self.freeze_VBN(True)

        # do rollout

        return 0

    def freeze_VBN(self, freeze):
        for m in self.modules():
            if isinstance(m, VirtualBatchNorm):
                m.update = not freeze

    def initialise_parameters(self, m):
        """
        Initialises the module parameters
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            nn.init.uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)

    def get_flat(self):
        """
        Returns the flattend vector of all trainable parameters
        Performance might be improved by not concatenating, but sending a flat view
        of each module parameter and then when perturbing with noise in ES to
        allocate noise slices across the different values. Maybe costly memory copying
        operations might be avoided.
        """
        flat_parameters = []
        for m in self.modules():
            if m.training and not isinstance(m, Policy) and not isinstance(m, nn.Sequential):
                for p in m.parameters():
                    flat_parameters.append(p.data.view(-1))

        return np.concatenate(flat_parameters)

    def set_from_flat(self, flat_parameters):
        start = 0
        for m in self.modules():
            if m.training and not isinstance(m, Policy) and not isinstance(m, nn.Sequential):
                for p in m.parameters():
                    size = np.prod(p.data.shape)
                    p.data = torch.tensor(flat_parameters[start:start + size]).view(p.data.shape)
                    start += size

    def set_ref_batch(self, ref_batch):
        self.ref_list = []
        self.ref_list.append(ref_batch)

    @property
    def needs_ref_batch(self):
        return True


def get_env():
    import gym
    from atari_wrappers import wrap_env
    env = gym.make("FrostbiteNoFrameskip-v4")
    return wrap_env(env)
