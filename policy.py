import numpy as np
import torch
import torch.nn as nn
from evo_utils import random_slices

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


Deepmind 2015 Architecture (+VirtualBatchNorm)

        conv1_out = conv_output(input_shape[-2], 8, 4)
        conv2_out = conv_output(conv1_out, 4, 2)
        conv3_out = conv_output(conv2_out, 3, 1)
        self.conv = nn.Sequential(nn.Conv2d(in_channels=input_shape[-1], out_channels=32,
                                            kernel_size=8, stride=4, bias=False),
                                  VirtualBatchNorm(input_shape=[32, conv1_out, conv1_out]),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=32, out_channels=64,
                                            kernel_size=4, stride=2, bias=False),
                                  VirtualBatchNorm(input_shape=[64, conv2_out, conv2_out]),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=64, out_channels=64,
                                            kernel_size=3, stride=1, bias=False),
                                  VirtualBatchNorm(input_shape=[64, conv3_out, conv3_out]),
                                  nn.ReLU())

        self.lin_dim = (conv3_out)**2 * 64

        self.mlp = nn.Sequential(nn.Linear(in_features=self.lin_dim, out_features=512, bias=False),
                                 VirtualBatchNorm(input_shape=[512]),
                                 nn.ReLU(),
                                 nn.Linear(in_features=512, out_features=output_shape))


"""

modes = ['last_layer', 'cnns_and_last_linear', 'all_except_first_linear', 'all_linear', 'all_except_linear', 'all_cnns', 'all_except_VBN', 'all_except_cnns', 'all']


class Policy(nn.Module):
    def __init__(self, input_shape, output_shape, ref_batch=None, big_net=False):
        super(Policy, self).__init__()

        self.stochastic_activation = True
        self.gain = 1.0
        self.optimize = 'last_layer'
        self.n_actions = output_shape

        self.rng = np.random.RandomState(0)
        self.slices = None

        def conv_output(width, kernel, stride, padding=0):
            return int((width - kernel + 2 * padding) // stride + 1)

        if big_net:  # Deepmind 2015 Architecture (+VirtualBatchNorm+one additional linear layer)
            conv1_out = conv_output(input_shape[-2], 8, 4)
            conv2_out = conv_output(conv1_out, 4, 2)
            conv3_out = conv_output(conv2_out, 3, 1)
            self.conv = nn.Sequential(nn.Conv2d(in_channels=input_shape[-1], out_channels=32,
                                                kernel_size=8, stride=4, bias=False),
                                      VirtualBatchNorm(input_shape=[32, conv1_out, conv1_out]),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels=32, out_channels=64,
                                                kernel_size=4, stride=2, bias=False),
                                      VirtualBatchNorm(input_shape=[64, conv2_out, conv2_out]),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels=64, out_channels=64,
                                                kernel_size=3, stride=1, bias=False),
                                      VirtualBatchNorm(input_shape=[64, conv3_out, conv3_out]),
                                      nn.ReLU())

            self.lin_dim = (conv3_out)**2 * 64

            self.mlp = nn.Sequential(nn.Linear(in_features=self.lin_dim, out_features=512, bias=False),
                                     VirtualBatchNorm(input_shape=[512]),
                                     nn.ReLU(),
                                     nn.Linear(in_features=512, out_features=256, bias=False),
                                     VirtualBatchNorm(input_shape=[256]),
                                     nn.ReLU(),
                                     nn.Linear(in_features=256, out_features=output_shape))
        else:
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
                # return np.random.randint(self.n_actions)
        else:
            return torch.argmax(x, dim=1)

    def rollout(self, theta, env, timestep_limit, max_runs=5, rank=None, novelty=False, render=False, curr_best=None):
        """
        rollout of the policy in the given env for no more than max_runs and no more than timestep_limit steps
        """
        # update policy
        self.set_from_flat(theta)
        # memory_clear_mask = self.rng.choice(int(0.1 * timestep_limit), size=256)

        t, e = 0, 1
        rewards = []

        roll_obs = []
        novelty_vector = []

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
                if novelty:
                    novelty_vector.append(env.unwrapped._get_ram())
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
        if novelty:
            return mean_reward, (roll_obs, novelty_vector)
        else:
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
                    elif t >= 5000:
                        all_rews.append(rewards)
                        print("Stopped rollout after {} game steps with reward {}.".format(t, rewards))
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
                if not isinstance(m, Policy) and not isinstance(m, nn.Sequential) and not isinstance(m, nn.Conv2d):
                    for p in m.parameters():
                        flat_parameters.append(p.data.view(-1))
            # only return 1/500 of the CNN parameters
            conv_parameters = []
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    for p in m.parameters():
                        conv_parameters.append(p.data.view(-1))
            flat_parameters.extend(conv_parameters)
            # conv_parameters = np.concatenate(conv_parameters)
            # # self.slices = random_slices(self.rng, len(conv_parameters), int(len(conv_parameters) / 500), 1)
            # # flat_parameters.append(conv_parameters[tuple(self.slices)])
            # self.slices = self.rng.randint(len(conv_parameters))
            # flat_parameters.append(np.array([conv_parameters[self.slices]], dtype=np.float32))

        elif self.optimize == 'all_except_linear':
            for m in self.modules():
                if not isinstance(m, Policy) and not isinstance(m, nn.Sequential) and not isinstance(m, nn.Linear):
                    for p in m.parameters():
                        flat_parameters.append(p.data.view(-1))
        elif self.optimize == 'all_except_VBN':
            for m in self.modules():
                if not isinstance(m, Policy) and not isinstance(m, nn.Sequential) and not isinstance(m, VirtualBatchNorm):
                    for p in m.parameters():
                        flat_parameters.append(p.data.view(-1))
        elif self.optimize == 'all_except_cnns':
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
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    for p in m.parameters():
                        flat_parameters.append(p.data.view(-1))
        elif self.optimize == 'cnns_and_last_linear':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    for p in m.parameters():
                        flat_parameters.append(p.data.view(-1))
            m = self.mlp[-1]
            for p in m.parameters():
                flat_parameters.append(p.data.view(-1))
        elif self.optimize == 'all_linear':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    for p in m.parameters():
                        flat_parameters.append(p.data.view(-1))
        elif self.optimize == 'last_layer':
            m = self.mlp[-1]
            for p in m.parameters():
                flat_parameters.append(p.data.view(-1))
        else:
            print("{} is not an allowed policy optimization mode!".format(self.optimize))

        return np.concatenate(flat_parameters)

    def set_from_flat(self, flat_parameters):
        start = 0
        if self.optimize == 'all':
            for m in self.modules():
                if not isinstance(m, Policy) and not isinstance(m, nn.Sequential) and not isinstance(m, nn.Conv2d):
                    for p in m.parameters():
                        size = np.prod(p.data.shape)
                        p.data = torch.tensor(flat_parameters[start:start + size]).view(p.data.shape)
                        start += size
            # set the 1/500 of the CNN parameters, which have been returned earlier
            # conv_parameters = []
            # for m in self.modules():
            #     if isinstance(m, nn.Conv2d):
            #         for p in m.parameters():
            #             conv_parameters.append(p.data.view(-1))
            # conv_parameters = np.concatenate(conv_parameters)
            # # size = [s.indices(len(conv_parameters)) for s in self.slices]
            # # size = int(np.sum([(s[1] - s[0]) / s[2] for s in size]))
            # size = 1
            # # add flat parameters to the flattened CNN parameters at the position of slices
            # # conv_parameters[tuple(self.slices)] = flat_parameters[start:start + size]
            # conv_parameters[self.slices] = flat_parameters[start:start + size]
            # # set the now changed conv_parameters to the actual network weights (to make sure)
            # start = 0
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    for p in m.parameters():
                        size = np.prod(p.data.shape)
                        # p.data = torch.tensor(conv_parameters[start:start + size]).view(p.data.shape)
                        p.data = torch.tensor(flat_parameters[start:start + size]).view(p.data.shape)
                        start += size

        elif self.optimize == 'all_except_linear':
            for m in self.modules():
                if not isinstance(m, Policy) and not isinstance(m, nn.Sequential) and not isinstance(m, nn.Linear):
                    for p in m.parameters():
                        size = np.prod(p.data.shape)
                        p.data = torch.tensor(flat_parameters[start:start + size]).view(p.data.shape)
                        start += size
        elif self.optimize == 'all_except_VBN':
            for m in self.modules():
                if not isinstance(m, Policy) and not isinstance(m, nn.Sequential) and not isinstance(m, VirtualBatchNorm):
                    for p in m.parameters():
                        size = np.prod(p.data.shape)
                        p.data = torch.tensor(flat_parameters[start:start + size]).view(p.data.shape)
                        start += size
        elif self.optimize == 'all_except_cnns':
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
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    for p in m.parameters():
                        size = np.prod(p.data.shape)
                        p.data = torch.tensor(flat_parameters[start:start + size]).view(p.data.shape)
                        start += size
        elif self.optimize == 'cnns_and_last_linear':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    for p in m.parameters():
                        size = np.prod(p.data.shape)
                        p.data = torch.tensor(flat_parameters[start:start + size]).view(p.data.shape)
                        start += size
            m = self.mlp[-1]
            for p in m.parameters():
                size = np.prod(p.data.shape)
                p.data = torch.tensor(flat_parameters[start:start + size]).view(p.data.shape)
                start += size
        elif self.optimize == 'all_linear':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    for p in m.parameters():
                        size = np.prod(p.data.shape)
                        p.data = torch.tensor(flat_parameters[start:start + size]).view(p.data.shape)
                        start += size
        elif self.optimize == 'last_layer':
            m = self.mlp[-1]
            for p in m.parameters():
                size = np.prod(p.data.shape)
                p.data = torch.tensor(flat_parameters[start:start + size]).view(p.data.shape)
                start += size
        else:
            print("{} is not an allowed policy optimization mode!".format(self.optimize))

    @property
    def num_parameters(self):
        n = 0
        for m in self.modules():
            if not isinstance(m, Policy) and not isinstance(m, nn.Sequential):
                for p in m.parameters():
                    n += np.prod(p.data.size())
        return n

    @property
    def theta(self):
        flat_parameters = []
        if self.optimize == 'all':
            for m in self.modules():
                if not isinstance(m, Policy) and not isinstance(m, nn.Sequential):
                    for p in m.parameters():
                        flat_parameters.append(p.data.view(-1))
        return np.concatenate(flat_parameters)


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




def f(pol, mode):
    if mode == 'all':
        for m in pol.modules():
            if not isinstance(m, Policy) and not isinstance(m, nn.Sequential):
                print(m)
    if mode == 'all_except_linear':
        for m in pol.modules():
            if not isinstance(m, Policy) and not isinstance(m, nn.Sequential) and not isinstance(m, nn.Linear):
                print(m)
    if mode == 'all_except_VBN':
        for m in pol.modules():
            if not isinstance(m, Policy) and not isinstance(m, nn.Sequential) and not isinstance(m, VirtualBatchNorm):
                print(m)
    if mode == 'all_except_cnns':
        for m in pol.modules():
            if not isinstance(m, Policy) and not isinstance(m, nn.Sequential) and not isinstance(m, nn.Conv2d):
                print(m)
    elif mode == 'all_except_first_linear':
        for m in pol.conv:
            print(m)
        for m in pol.mlp[1:]:
            print(m)
    elif mode == 'all_cnns':
        for m in pol.modules():
            if isinstance(m, nn.Conv2d):
                print(m)
    elif mode == 'cnns_and_last_linear':
        for m in pol.modules():
            if isinstance(m, nn.Conv2d):
                print(m)
        m = pol.mlp[-1]
        print(m)
    elif mode == 'all_linear':
        for m in pol.modules():
            if isinstance(m, nn.Linear):
                print(m)
    elif mode == 'last_layer':
        m = pol.mlp[-1]
        print(m)


"""
