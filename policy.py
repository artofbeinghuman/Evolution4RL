import numpy as np
import torch
import torch.nn as nn
import json


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
        var = ((x - mean)**2).mean(0, keepdim=True)
        return mean, var

    def forward(self, x):
        if self.update:
            mean, var = self.get_stats(x)
            if self.ref_mean is not None and self.ref_var is not None:
                self.batch_size += x.shape[0]
                coeff = x.shape[0] / self.batch_size
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


"""
# slim network
        conv1_out = conv_output(input_shape[-2], 8, 4)
        conv2_out = conv_output(conv1_out, 4, 2)
        conv3_out = conv_output(conv2_out, 3, 2)
        self.conv = nn.Sequential(nn.Conv2d(in_channels=input_shape[-1], out_channels=16,
                                            kernel_size=8, stride=4),
                                  VirtualBatchNorm(input_shape=[16, conv1_out, conv1_out]),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=16, out_channels=32,
                                            kernel_size=4, stride=2),
                                  VirtualBatchNorm(input_shape=[32, conv2_out, conv2_out]),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=32, out_channels=16,
                                            kernel_size=4, stride=2),
                                  VirtualBatchNorm(input_shape=[16, conv3_out, conv3_out]),
                                  nn.ReLU())

        self.lin_dim = (conv3_out)**2 * 32

        self.mlp = nn.Sequential(nn.Linear(in_features=self.lin_dim, out_features=256),
                                 VirtualBatchNorm(input_shape=[256]),
                                 nn.ReLU(),
                                 nn.Linear(in_features=256, out_features=64),
                                 VirtualBatchNorm(input_shape=[64]),
                                 nn.ReLU(),
                                 nn.Linear(in_features=64, out_features=output_shape))


"""

modes = ['last_layer', 'cnns_and_last_linear', 'all_except_first_linear', 'all_linear', 'all_except_linear', 'all_cnns', 'all_except_VBN', 'all_except_cnns', 'all']


class Policy(nn.Module):
    def __init__(self, input_shape, output_shape, ref_batch=None):
        super(Policy, self).__init__()

        self.stochastic_activation = True
        self.gain = 1.0
        self.optimize = 'last_layer'

        def conv_output(width, kernel, stride, padding=0):
            return int((width - kernel + 2 * padding) // stride + 1)

        conv1_out = conv_output(input_shape[-2], 8, 4)
        conv2_out = conv_output(conv1_out, 4, 2)
        self.conv = nn.Sequential(nn.Conv2d(in_channels=input_shape[-1], out_channels=16,
                                            kernel_size=8, stride=4, bias=False),
                                  VirtualBatchNorm(input_shape=[16, conv1_out, conv1_out]),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=16, out_channels=32,
                                            kernel_size=4, stride=2, bias=False),
                                  VirtualBatchNorm(input_shape=[32, conv2_out, conv2_out]),
                                  nn.ReLU())

        self.lin_dim = (conv2_out)**2 * 32

        self.mlp = nn.Sequential(nn.Linear(in_features=self.lin_dim, out_features=256, bias=False),
                                 VirtualBatchNorm(input_shape=[256]),
                                 nn.ReLU(),
                                 nn.Linear(in_features=256, out_features=output_shape))

        self.apply(self.initialise_parameters)
        self.initialise_VBN(ref_batch)
        self.eval()  # no gradients

    def forward(self, obs):
        # expect an input of form: N x C x H x W
        # where N should be 1 (except for ref_batch), H=W=84 as per atari_wrappers.wrap_deepmind() and C=4 due to framestacking of 4 greyscale frames
        x = self.conv(obs)
        x = x.view(-1, self.lin_dim)
        x = self.mlp(x)
        return self.activation(x)

    def activation(self, x):
        if self.stochastic_activation:
            if np.random.rand() < 0.1:
                x = nn.functional.softmax(x, dim=-1)
                return torch.distributions.categorical.Categorical(probs=x).sample()
            else:
                return torch.argmax(x, dim=1)
        else:
            return torch.argmax(x, dim=1)

    def rollout(self, theta, env, timestep_limit, max_runs=5, rank=None, render=False, curr_best=None):
        """
        rollout of the policy in the given env for no more than max_runs and no more than timestep_limit steps
        """
        # update policy
        self.set_from_flat(theta)
        # memory_clear_mask = self.rng.choice(int(0.1 * timestep_limit), size=256)

        t, e = 0, 1
        rewards = []

        roll_obs = []

        # do rollouts until max_runs rollouts where done or
        # until time left is less than 70% of mean rollout length
        while e <= max_runs and (timestep_limit - t) >= 0.7 * t / e:
            if t == 0:
                e = 0  # only start with 0 here to avoid issue in condition check for first loop
            e += 1

            # do rollout
            rewards.append([])
            obs = env.reset()
            while t <= timestep_limit:
                action = int(self.forward(to_obs_tensor(obs)))
                obs, rew, done, _ = env.step(action)
                rewards[-1].append(rew)

                if curr_best:
                    roll_obs.append(obs)
                if render:
                    env.render()

                t += 1
                if done:
                    break

        if render:
            env.close()

        rewards = [np.sum(rews) for rews in rewards]
        mean_reward = np.mean(rewards)
        return mean_reward, roll_obs

    def play(self, env, theta=None, loop=False, save=False):
        if theta is not None:
            self.set_from_flat(theta)

        self.eval()

        t, rewards = 0, 0
        all_rews = []
        all_obs = []
        obs = env.reset()
        done = False
        if save:
            for _ in range(5):
                while True:
                    action = int(self.forward(to_obs_tensor(obs)))
                    obs, rew, done, _ = env.step(action)
                    rewards += rew
                    all_obs.append(obs)
                    t += 1
                    if done:
                        all_rews.append(rewards)
                        print("Died after {} game steps with reward {}.".format(t, rewards))
                        t, rewards = 0, 0
                        obs = env.reset()
                        break
        else:
            try:
                import time
                while not done or loop:
                    time.sleep(0.02)
                    action = int(self.forward(to_obs_tensor(obs)))
                    obs, rew, done, _ = env.step(action)
                    rewards += rew
                    env.render()
                    t += 1
                    if done:
                        all_rews.append(rewards)
                        print("Died after {} game steps with reward {}.".format(t, rewards))
                        t, rewards = 0, 0
                        obs = env.reset()
                env.close()
            except KeyboardInterrupt:
                env.close()
                print("Game stopped by user, total mean reward after {} runs: {:.2f}".format(len(all_rews), np.mean(all_rews)))
        return all_obs, all_rews

    def initialise_parameters(self, m):
        """
        Initialises the module parameters
        http://archive.is/EGwsP
        """

        if isinstance(m, nn.Linear):
            # nn.init.normal_(m.weight.data, mean=0, std=0.01)
            nn.init.xavier_uniform_(m.weight.data, gain=self.gain)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Conv2d):
            # nn.init.normal_(m.weight.data, mean=0, std=0.01)
            nn.init.xavier_normal_(m.weight.data, gain=self.gain)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)

    def freeze_VBN(self, freeze):
        for m in self.modules():
            if isinstance(m, VirtualBatchNorm):
                m.update = not freeze

    def initialise_VBN(self, ref_batch):
        self.freeze_VBN(False)
        if ref_batch is not None:
            self.forward(ref_batch)
        else:
            print("Generate ref_batch and manually initialise VirtualBatchNorm!")
        self.freeze_VBN(True)

    def get_flat(self):
        """
        Returns the flattend vector of all trainable parameters
        Performance might be improved by not concatenating, but sending a flat view
        of each module parameter and then when perturbing with noise in ES to
        allocate noise slices across the different values. Maybe costly memory copying
        operations might be avoided.
        """
        flat_parameters = []
        if self.optimize == 'all':
            for m in self.modules():
                if not isinstance(m, Policy) and not isinstance(m, nn.Sequential):
                    for p in m.parameters():
                        flat_parameters.append(p.data.view(-1))
        if self.optimize == 'all_except_linear':
            for m in self.modules():
                if not isinstance(m, Policy) and not isinstance(m, nn.Sequential) and not isinstance(m, nn.Linear):
                    for p in m.parameters():
                        flat_parameters.append(p.data.view(-1))
        if self.optimize == 'all_except_VBN':
            for m in self.modules():
                if not isinstance(m, Policy) and not isinstance(m, nn.Sequential) and not isinstance(m, VirtualBatchNorm):
                    for p in m.parameters():
                        flat_parameters.append(p.data.view(-1))
        if self.optimize == 'all_except_cnns':
            for m in self.modules():
                if not isinstance(m, Policy) and not isinstance(m, nn.Sequential) and not isinstance(m, nn.Conv2d):
                    for p in m.parameters():
                        flat_parameters.append(p.data.view(-1))
        elif self.optimize == 'all_except_first_linear':
            for m in self.conv:
                for p in m.parameters():
                    flat_parameters.append(p.data.view(-1))
            for m in self.mlp[1:]:
                for p in m.parameters():
                    flat_parameters.append(p.data.view(-1))
        elif self.optimize == 'all_cnns':
            for m in [self.conv[0], self.conv[3]]:
                for p in m.parameters():
                    flat_parameters.append(p.data.view(-1))
        elif self.optimize == 'cnns_and_last_linear':
            for m in [self.conv[0], self.conv[3], self.mlp[3]]:
                for p in m.parameters():
                    flat_parameters.append(p.data.view(-1))
        elif self.optimize == 'all_linear':
            for m in [self.mlp[0], self.mlp[3]]:
                for p in m.parameters():
                    flat_parameters.append(p.data.view(-1))
        elif self.optimize == 'last_layer':
            for p in self.mlp[3].parameters():
                flat_parameters.append(p.data.view(-1))

        return np.concatenate(flat_parameters)

    def set_from_flat(self, flat_parameters):
        start = 0
        if self.optimize == 'all':
            for m in self.modules():
                if not isinstance(m, Policy) and not isinstance(m, nn.Sequential):
                    for p in m.parameters():
                        size = np.prod(p.data.shape)
                        p.data = torch.tensor(flat_parameters[start:start + size]).view(p.data.shape)
                        start += size
        if self.optimize == 'all_except_linear':
            for m in self.modules():
                if not isinstance(m, Policy) and not isinstance(m, nn.Sequential) and not isinstance(m, nn.Linear):
                    for p in m.parameters():
                        size = np.prod(p.data.shape)
                        p.data = torch.tensor(flat_parameters[start:start + size]).view(p.data.shape)
                        start += size
        if self.optimize == 'all_except_VBN':
            for m in self.modules():
                if not isinstance(m, Policy) and not isinstance(m, nn.Sequential) and not isinstance(m, VirtualBatchNorm):
                    for p in m.parameters():
                        size = np.prod(p.data.shape)
                        p.data = torch.tensor(flat_parameters[start:start + size]).view(p.data.shape)
                        start += size
        if self.optimize == 'all_except_cnns':
            for m in self.modules():
                if not isinstance(m, Policy) and not isinstance(m, nn.Sequential) and not isinstance(m, nn.Conv2d):
                    for p in m.parameters():
                        size = np.prod(p.data.shape)
                        p.data = torch.tensor(flat_parameters[start:start + size]).view(p.data.shape)
                        start += size
        elif self.optimize == 'all_except_first_linear':
            for m in self.conv:
                for p in m.parameters():
                    size = np.prod(p.data.shape)
                    p.data = torch.tensor(flat_parameters[start:start + size]).view(p.data.shape)
                    start += size
            for m in self.mlp[1:]:
                for p in m.parameters():
                    size = np.prod(p.data.shape)
                    p.data = torch.tensor(flat_parameters[start:start + size]).view(p.data.shape)
                    start += size
        elif self.optimize == 'all_cnns':
            for m in [self.conv[0], self.conv[3]]:
                for p in m.parameters():
                    size = np.prod(p.data.shape)
                    p.data = torch.tensor(flat_parameters[start:start + size]).view(p.data.shape)
                    start += size
        elif self.optimize == 'cnns_and_last_linear':
            for m in [self.conv[0], self.conv[3], self.mlp[3]]:
                for p in m.parameters():
                    size = np.prod(p.data.shape)
                    p.data = torch.tensor(flat_parameters[start:start + size]).view(p.data.shape)
                    start += size
        elif self.optimize == 'all_linear':
            for m in [self.mlp[0], self.mlp[3]]:
                for p in m.parameters():
                    size = np.prod(p.data.shape)
                    p.data = torch.tensor(flat_parameters[start:start + size]).view(p.data.shape)
                    start += size
        elif self.optimize == 'last_layer':
            for p in self.mlp[3].parameters():
                size = np.prod(p.data.shape)
                p.data = torch.tensor(flat_parameters[start:start + size]).view(p.data.shape)
                start += size

    def set_ref_batch(self, ref_batch):
        self.ref_list = []
        self.ref_list.append(ref_batch)

    @property
    def num_parameters(self):
        n = 0
        for m in self.modules():
            if not isinstance(m, Policy) and not isinstance(m, nn.Sequential):
                for p in m.parameters():
                    n += np.prod(p.data.size())
        return n

    @property
    def needs_ref_batch(self):
        return True


def get_env():
    import gym
    from gym_wrappers import wrap_env
    env = gym.make("SeaquestNoFrameskip-v4")
    return wrap_env(env)


def get_pol():
    env = get_env()
    pol = Policy(env.observation_space.shape, env.action_space.n)
    pol.freeze_VBN(False)
    pol._ref_batch = get_ref_batch(env)
    pol.forward(pol._ref_batch)
    pol.freeze_VBN(True)
    return env, pol


def print_paras(pol):
    s = []
    total = 0
    for seq in [pol.conv, pol.mlp]:
        for m in seq:
            paras = 0
            for p in m.parameters():
                paras += np.prod(p.shape)
            s.append([m, paras])
            total += paras
    print([[t[0], t[1], t[1] / total * 100] for t in s])


def show_ob(ob):
    import matplotlib.pyplot as plt
    ob = np.squeeze(np.array(ob))
    if ob.shape[-1] <= 4:
        for i in range(ob.shape[-1]):
            plt.subplot(2, 2, i + 1)
            plt.axis('off')
            plt.imshow(ob[:, :, i], cmap='gray', aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


"""

def act(obs):
    # expect an input of form: N x C x H x W
    # where N should be 1 (except for ref_batch), H=W=84 as per atari_wrappers.wrap_deepmind() and C=4 due to framestacking of 4 greyscale frames
    x = pol.conv(obs)
    x = x.view(-1, pol.lin_dim)
    return pol.mlp(x)


def test(ob=None, a=False):
    if a:
        pol.apply(initialise_parameters)
        pol.eval()
    print(pol.conv[0].weight.max(), pol.conv[0].weight.min())
    print(pol.conv[3].weight.max(), pol.conv[3].weight.min())
    print(pol.mlp[0].weight.max(), pol.mlp[0].weight.min())
    print(pol.mlp[0].weight.max(), pol.mlp[0].weight.min())
    if ob is None:
        ob = to_obs_tensor(env.reset())
    print(act(ob))
    print(pol.forward(ob))


"""
