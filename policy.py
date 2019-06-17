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
    rng = np.random.RandomState(0)
    while len(ref_batch) < batch_size:
        ob, _, done, _ = env.step(env.action_space.sample())
        if done:
            ob = env.reset()
        elif rng.rand() < p:
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
    with affine transformation (gamma, beta learned via ES)
    """

    def __init__(self, input_shape):
        super(VirtualBatchNorm, self).__init__()
        # batch statistics
        self.eps = 1e-5  # epsilon
        self.ref_mean = None
        self.ref_var = None
        self.gamma = nn.parameter.Parameter(torch.ones([1] + input_shape))
        self.beta = nn.parameter.Parameter(torch.zeros([1] + input_shape))

        self.update = True
        self.batch_size = 1

        self.repr = '{}(input_shape={})'.format(self.__class__.__name__, input_shape)

    def get_stats(self, x):
        mean = x.mean(0, keepdim=True)
        var = (x ** 2).mean(0, keepdim=True) - mean**2
        return mean, var

    def forward(self, x):
        if self.update:
            mean, var = self.get_stats(x)
            if self.ref_mean is not None and self.ref_var is not None:
                coeff = x.shape[0] / (self.batch_size + x.shape[0])
                self.ref_mean = coeff * mean + (1 - coeff) * self.ref_mean
                self.ref_var = coeff * var + (1 - coeff) * self.ref_var
            else:
                self.ref_mean, self.ref_var = mean, var
            return self._normalize(x)
        else:
            return self._normalize(x)

    def _normalize(self, x):
        return (x - self.ref_mean) / torch.sqrt(self.ref_var + self.eps) * self.gamma + self.beta

    def __repr__(self):
        return (self.repr)


class Policy(nn.Module):
    def __init__(self, input_shape, output_shape, rng=np.random.RandomState(0)):
        super(Policy, self).__init__()

        #self.env = env
        self.rng = rng
        self._ref_batch = None
        #input_shape = self.env.observation_space.shape
        #output_shape = self.env.action_space.n

        def conv_output(width, kernel, stride, padding=0):
            return int((width - kernel + 2 * padding) // stride + 1)
        conv1_out = conv_output(input_shape[-2], 8, 4)
        conv2_out = conv_output(conv1_out, 4, 2)
        self.conv = nn.Sequential(nn.Conv2d(in_channels=input_shape[-1], out_channels=16,
                                            kernel_size=8, stride=4),
                                  VirtualBatchNorm(input_shape=[16, conv1_out, conv1_out]),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=16, out_channels=32,
                                            kernel_size=4, stride=2),
                                  VirtualBatchNorm(input_shape=[32, conv2_out, conv2_out]),
                                  nn.ReLU())

        self.lin_dim = (conv2_out)**2 * 32

        self.mlp = nn.Sequential(nn.Linear(in_features=self.lin_dim, out_features=256),
                                 VirtualBatchNorm(input_shape=[256]),
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

    def rollout(self, theta, env, timestep_limit, max_runs=5, novelty=False, rank=None):
        """
        rollout of the policy in the given env for no more than max_runs and no more than timestep_limit steps
        """
        # update policy
        self.set_from_flat(theta)

        t, e = 0, 1
        rewards, novelty_vector = 0, []

        # do rollouts until max_runs rollouts where done or
        # until time left is less than 70% of mean rollout length
        while e <= max_runs and (timestep_limit - t) >= 0.7 * t / e:
            if t == 0:
                e = 0  # only start with 0 here to avoid issue in condition check for first loop
            e += 1
            print("{}, run {}, timestep {}".format(rank, e, t))
            # initialise or update virtual batch normalization
            self.freeze_VBN(False)
            if self._ref_batch is None:
                self._ref_batch = get_ref_batch(env)
            else:  # suppose you have a list from the last run
                # I figured it is faster to choose out of all obs of the last run,
                # instead of flipping a coin after every observation
                self._ref_batch = np.array(self._ref_batch)[self.rng.choice(int(len(self._ref_batch)), 64)]
                self._ref_batch = torch.tensor(self._ref_batch.transpose(0, 3, 1, 2))
            self.forward(self._ref_batch)
            self.freeze_VBN(True)
            self._ref_batch = []

            # do rollout
            obs = env.reset()
            for _ in range(timestep_limit - t):
                action = self.forward(to_obs_tensor(obs))
                obs, rew, done, _ = env.step(action)
                rewards += rew
                self._ref_batch.append(obs)
                if novelty:
                    novelty_vector.append(env.unwrapped._get_ram())

                t += 1
                if done:
                    break

        mean_reward = rewards / e
        # return mean_reward, novelty_vector
        return mean_reward

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
    from gym_wrappers import wrap_env
    env = gym.make("FrostbiteNoFrameskip-v4")
    return wrap_env(env)
