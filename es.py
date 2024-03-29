import numpy as np
from functools import partial
import torch
from policy import Policy, get_ref_batch
from optimizers import SGD, Adam
import time
from evo_utils import *
import socket

# Credit goes to Nathaniel Rodriguez from whom I adopted large parts of this code
# https://github.com/Nathaniel-Rodriguez/evostrat/blob/master/evostrat/evostrat.py


def log(es, msg):
    if es._rank == 0:
        if es.log is not None:
            es.log.write("\n" + msg)
            es.log.flush()
            print(msg)


def save_video(obs, path):
    import imageio
    import warnings
    from skimage import img_as_ubyte
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imageio.mimwrite(path, img_as_ubyte(np.array(obs)), fps=30)


def behaviour_distance(x, y):

    # bring both behaviours up to same length
    n, m = len(x), len(y)
    if n > m:
        a = np.linalg.norm(y - x[:m])
        b = np.linalg.norm(y[-1] - x[m:])
    else:
        a = np.linalg.norm(x - y[:n])
        b = np.linalg.norm(x[-1] - y[n:])
    return np.sqrt(a**2 + b**2)


def compute_novelty_vs_archive(archive, archive_idc, behaviour, k):
    distances = []
    nov = behaviour.astype(np.float)
    for indx in archive_idc:
        distances.append(behaviour_distance(archive[indx[0]:indx[1]].astype(np.float), nov))

    # Pick k nearest neighbors
    distances = np.array(distances)
    top_k_indicies = (distances).argsort()[:k]
    top_k = distances[top_k_indicies]
    return top_k.mean()


class ES:
    """
    Runs a basic distributed Evolutionary strategy using MPI. The centroid
    is updated via a log-rank weighted sum of the population members that are
    generated from a normal distribution. This class evaluates a reward function
    ie fitness function.

    Creates a large RN table
    Draws a proper basic slice (start,stop,step) from RN table
    Draws a proper basic slice (start,stop,step) from parameter list
    Applies RNGs from sliced table to sliced parameters.
    Then each rank only needs to reconstruct the slice draws, and not
    the full rng list, which makes it less expensive
    The two step slice draw helps smooth out the impending correlations between
    RNs in the RN table space when finally applied to parameter space.
    Becomes more of an issue the larger the slice is with respect to the
    parameter space and the table space. Worst case is small table with large
    slice and many parameters.
    max_table_step is the maximum stride that an iterator will take over the
    table.
    max_param_step is the maximum stride that an iterator will take over the
    parameters, when drawing a random subset for permutation. A larger step will
    help wash out correlations in randomness. However,if the step is too large
    you risk overlapping old values as it wraps around the arrays.
    WARNING: If the # mutations approaches # parameters, make sure that the
    max_param_step == 1, else overlapping will cause actual # mutations to be
    less than the desired value.
    Currently, the global_rng draws perturbed dimensions for each iteration.
    This could be changed to let workers draw their own dimensions to perturb.


    """

    def __init__(self, exp, **kwargs):
        """
        :param xo: initial centroid
        kwargs:
        :param sigma (optional): noise/perturbation variance to use for mutation (fixed and uniform variance of mutation), i.e. mutation strength
        :param num_mutations (optional): number of dims to mutate in each iter (defaults to #dim)
        :param seed (optional): global seed for all workers
        :param rand_num_table_size (optional): size of the random number table
        :param time_step_limit: maximum number of timesteps for each policy evaluation (one or more rollouts)
        :param max_runs_per_eval: maximum number of rollouts per policy evaluation
        :param use_novelty (optional): whether to employ novelty weighted updating
        :param OpenAIES (optional): whether to run the ES in the OpenAI style or traditionally (put optimizer in exp json!)
        :param num_parents (optional): for truncated selection, default for OpenAIES = all (=number of workers), for !OpenAIES = 0.5*number of workers

        """
        # Initiate MPI
        from mpi4py import MPI
        import socket
        self._MPI = MPI
        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()

        if self._size > 1:
            assert self._size % 2 == 0

        if self._rank == 0:
            self._path = kwargs.get("log_path", "No_path_specified")
            self.log = open(self._path + ".log", 'w+')
            log(self, kwargs.get("initial_text", "\n"))
        else:
            self._path = ""
        self._path = self._comm.bcast(self._path, root=0)

        self._global_seed = kwargs.get('seed', 123)
        torch.manual_seed(self._global_seed)
        self._global_rng = np.random.RandomState(self._global_seed)
        self._noise_seed = self._global_rng.choice(1000000, size=1)
        self._seed_set = self._global_rng.choice(1000000, size=int(np.ceil(self._size / 2)), replace=False)
        # for mirrored sampling, two subsequent workers (according to rank) will have the same seed/rng
        # the even ranked workers will perturb positively, the odd ones negatively
        self._seed_set = [s for ss in [[s, s] for s in self._seed_set] for s in ss]
        self._worker_rngs = [np.random.RandomState(seed) for seed in self._seed_set]
        # self._par_choices = np.arange(0, self._num_parameters, dtype=np.int32)
        self._generation_number = 0
        self._score_history = []

        # create policy of this worker
        # (parameters should be initialised randomly via global rng, such that all workers start with identical centroid)
        self.exp = exp
        self.env = get_env_from(exp)
        self._ref_batch = get_ref_batch(self.env, batch_size=2**10, p=0.2) if self._rank == 0 else torch.empty([2**10, 4, 84, 84])
        self._ref_batch = self._comm.bcast(self._ref_batch, root=0)
        self._big_net = kwargs.get('big_net', False)
        self.policy = Policy(self.env.observation_space.shape, self.env.action_space.n, ref_batch=self._ref_batch, big_net=self._big_net)
        self._policy_activation = kwargs.get('activation', False)
        self.policy.activation_mode = self._policy_activation
        self._optimize = kwargs.get('optimize', 'last_layer')
        self.policy.optimize = self._optimize

        # State
        self._update_best_flag = False
        self._theta = self.policy.get_flat()
        self._old_theta = self._theta.copy()
        self._running_best = self._theta.copy()
        self._running_best_reward = np.float32(np.finfo(np.float32).min)
        self.optimizer = {'sgd': SGD, 'adam': Adam}[exp['optimizer']['type']](self._old_theta, **exp['optimizer']['args'])
        log(self, "Optimizing {} out of {} network parameters ({:.2f}%).".format(int(self._theta.size), self.policy.num_parameters, 100 * int(self._theta.size) / self.policy.num_parameters))
        self._update_ratios = []

        # novelty search
        novelty = kwargs.get('novelty', False)
        self._archive = []
        self._rew_nov_tradeoff = 1.0

        # User input parameters
        self.objective = self.policy.rollout
        timestep_limit = exp['config']['timestep_limit']  # kwargs.get('timestep_limit', 10000)
        max_runs = exp['config']['max_runs_per_eval']  # kwargs.get('max_runs_per_eval', 5)
        render = kwargs.get('render', False) if self._rank == 0 else False
        self.obj_kwargs = {'env': self.env, 'timestep_limit': timestep_limit, 'max_runs': max_runs, 'rank': self._rank, 'render': render, 'novelty': novelty}
        self._sigma = np.float32(kwargs.get('sigma', 0.05))  # this is the noise_stdev parameter from Uber json-configs
        self._num_parameters = len(self._theta)
        self._mutate = kwargs.get('mutate', 1)
        self._num_mutations = int(self._num_parameters / self._mutate)  # limited perturbartion ES as in Zhang et al 2017, ch.4.1
        self._OpenAIES = not kwargs.get('classic_es', True)
        self._video = not kwargs.get('no_videos', True)

        if self._OpenAIES:
            # use centered ranks [0.5, -0.5]
            self._num_parents = int(self._size / kwargs.get('num_parents', 1))
            # self._num_parents = 1  # int(np.ceil(0.1 * self._size))
            self._weights = np.arange(self._size, 0, -1, dtype=np.float32)
            # self._weights = self._weights / np.sum(self._weights)
            self._weights = self._weights / np.array([self._size], dtype=np.float32)
            self._weights -= 0.5
            self._weights[self._size // 2] = 0.001  # to avoid divide by zero
            # multiply 1/sigma factor directly at this point,
            self._weights /= np.array([self._sigma], dtype=np.float32)
            self._weights = self._weights[:self._num_parents]
            self._weights.astype(np.float32, copy=False)
        else:
            # Classics ES:
            self._num_parents = int(self._size / kwargs.get('num_parents', 2))  # truncated selection
            # self._num_parents = 1  # int(np.ceil(0.1 * self._size))
            self._weights = np.log(self._num_parents + 0.5) - np.log(
                np.arange(1, self._num_parents + 1))
            self._weights = self._weights / np.sum(self._weights)
            self._weights.astype(np.float32, copy=False)
            self._update_ratios = ['-']

        # create communication groups which are local to each node.
        # Later generate random noise table on each node locally via these comm_local
        local_ip = socket.gethostname()
        all_ips = self._comm.gather(local_ip, root=0)
        if self._rank == 0:
            num = len(all_ips)
            all_ips = list(set(all_ips))
            log(self, "{} workers active on {}.".format(num, "{} node".format(len(all_ips)) if len(all_ips) == 1 else "{} nodes".format(len(all_ips))))
        all_ips = self._comm.bcast(all_ips, root=0)
        self._comm_local = self._comm.Split(color=all_ips.index(local_ip), key=self._rank)

        self._rand_num_table_size = kwargs.get("rand_num_table_size", 200000)  # 250000000 ~ 1GB of noise
        self._max_table_step = kwargs.get("max_table_step", 5)
        self._max_param_step = kwargs.get("max_param_step", 1)
        nbytes = self._rand_num_table_size * MPI.FLOAT.Get_size() if self._comm_local.rank == 0 else 0
        win = MPI.Win.Allocate_shared(nbytes, MPI.FLOAT.Get_size(), comm=self._comm_local)
        buf, itemsize = win.Shared_query(0)
        assert itemsize == MPI.FLOAT.Get_size()
        self._rand_num_table = np.ndarray(buffer=buf, dtype='f', shape=(self._rand_num_table_size,))

        # calculate random table distributed across workers on local node
        t = time.time()
        step = int(np.ceil(self._rand_num_table_size / self._comm_local.size)) if self._comm_local.rank == self._comm_local.size - 1 else int(np.floor(self._rand_num_table_size / self._comm_local.size))
        noise_rng = np.random.RandomState(self._noise_seed * (self._comm_local.rank + 1))
        if self._comm_local.rank == self._comm_local.size - 1:
            self._rand_num_table[-step:] = noise_rng.randn(step)
        else:
            self._rand_num_table[self._comm_local.rank * step:(self._comm_local.rank + 1) * step] = noise_rng.randn(step)
        self._comm_local.Barrier()
        if self._comm_local.rank == 0:
            # Fold step-size into noise
            self._rand_num_table *= self._sigma
            log(self, "Calculated Random Table in {}s.".format(time.time() - t))

        archive_length = self._rand_num_table_size // 100
        nbytes = archive_length * 128 * MPI.FLOAT.Get_size() if self._comm_local.rank == 0 else 0
        win = MPI.Win.Allocate_shared(nbytes, MPI.FLOAT.Get_size(), comm=self._comm_local)
        buf, itemsize = win.Shared_query(0)
        assert itemsize == MPI.FLOAT.Get_size()
        self._archive = np.ndarray(buffer=buf, dtype='f', shape=(archive_length, 128))
        self._archive_idc = []

        self._comm.Barrier()
        if self._rank % 96 < 2:
            print("## randtable on worker {} on node {}:".format(self._rank, socket.gethostname()), self._rand_num_table[:5])
        self._comm.Barrier()

    def __getstate__(self):

        state = {"exp": self.exp,
                 "obj_kwargs": self.obj_kwargs,
                 "_ref_batch": self._ref_batch,
                 "policy": self.policy,
                 "_path": self._path,
                 "_optimize": self._optimize,
                 "_noise_seed": self._noise_seed,
                 "_big_net": self._big_net,
                 # "_archive": self._archive,
                 "_rew_nov_tradeoff": self._rew_nov_tradeoff,
                 "_policy_activation": self._policy_activation,
                 "optimizer": self.optimizer,
                 "_OpenAIES": self._OpenAIES,
                 "_update_ratios": self._update_ratios,
                 "_sigma": self._sigma,
                 "_num_parameters": self._num_parameters,
                 "_num_mutations": self._num_mutations,
                 "_num_parents": self._num_parents,
                 "_global_seed": self._global_seed,
                 "_weights": self._weights,
                 "_global_rng": self._global_rng,
                 "_seed_set": self._seed_set,
                 "_worker_rngs": self._worker_rngs,
                 "_generation_number": self._generation_number,
                 "_score_history": self._score_history,
                 "_theta": self._theta,
                 "_running_best": self._running_best,
                 "_running_best_reward": self._running_best_reward,
                 "_update_best_flag": self._update_best_flag,
                 "_rand_num_table_size": self._rand_num_table_size,
                 "_max_table_step": self._max_table_step,
                 "_max_param_step": self._max_param_step}

        return state

    def __setstate__(self, state):

        for key in state:
            setattr(self, key, state[key])

        # Reconstruct larger structures and load MPI
        # self._par_choices = np.arange(0, self._num_parameters, dtype=np.int32)
        self._old_theta = self._theta.copy()
        from mpi4py import MPI
        self._MPI = MPI
        self._comm = MPI.COMM_WORLD
        self._size = self._comm.Get_size()
        self._rank = self._comm.Get_rank()
        torch.manual_seed(self._global_seed)

        self.env = self.obj_kwargs['env']  # get_env_from(self.exp)
        # self.policy = Policy(self.env.observation_space.shape, self.env.action_space.n, self._ref_batch)
        # self.policy.stochastic_activation = self._policy_activation
        # self.policy.set_from_flat(self._running_best)
        self.objective = self.policy.rollout

        # won't load huge noise table on local laptop
        if False:
            nbytes = self._rand_num_table_size * MPI.FLOAT.Get_size() if self._rank == 0 else 0
            win = MPI.Win.Allocate_shared(nbytes, MPI.FLOAT.Get_size(), comm=self._comm)
            buf, itemsize = win.Shared_query(0)
            assert itemsize == MPI.FLOAT.Get_size()
            self._rand_num_table = np.ndarray(buffer=buf, dtype='f', shape=(self._rand_num_table_size,))
            if self._rank == 0:
                t = time.time()
                noise_rng = np.random.RandomState(self._noise_seed)
                self._rand_num_table[:] = noise_rng.randn(self._rand_num_table_size)
                log(self, "Calculated Random Table in {}s".format(time.time() - t))
                # Fold step-size into table values
                self._rand_num_table *= self._sigma

        self._comm.Barrier()

    def __call__(self, num_generations):
        """
        :param num_generations: how many generations it will run for
        :return: None
        """
        if socket.gethostname() == "mlpad":
            recent_gens = 3
        else:
            recent_gens = 15
        w = get_weight(recent_gens)
        t = time.time()
        tt = time.time()
        partial_objective = partial(self.objective, **self.obj_kwargs)
        for i in range(num_generations):
            self._comm.Barrier()
            self._generation_number += 1

            # adapt novelty reward tradeoff parameter each recent_gens generations
            if len(self._score_history) >= recent_gens and self._generation_number % recent_gens == 1:
                # last_rewards = [np.mean(rews) for rews in self._score_history[-recent_gens:]]
                last_rewards = [np.max(rews) for rews in self._score_history[-recent_gens:]]
                # last_rewards = [np.mean(np.argsort(rews)[::-1][:int(np.ceil(self._size * 0.1))]) for rews in self._score_history[-recent_gens:]]
                # if reward is increasing, optimize for rewards and decrease sigma, else for novelty and increase sigma
                if np.dot(w, last_rewards) > 0.1 * np.mean(last_rewards):
                    self._set_sigma(self._sigma / 2)
                    self._set_rew_nov_tradeoff(self._rew_nov_tradeoff + 0.1)
                else:
                    self._set_sigma(self._sigma * 2)
                    self._set_rew_nov_tradeoff(self._rew_nov_tradeoff - 0.1)

            if self.policy.optimize == 'all':
                log(self, "Optimizing index {} of CNN parameters".format(self.policy.slices))

            self._comm.Barrier()
            self._update(partial_objective)
            log(self, "Gen {} took {}s.".format(self._generation_number, time.time() - t))
            t = time.time()
        log(self, "\nFinished run in " + get_hms_string(time.time() - tt) + " with total best {:.2f}.\n".format(self._running_best_reward))
        if self._rank == 0:
            if self._video:
                self.showcase(save=True)
            self.log.close()

    def _draw_random_table_slices(self, rng):
        """
        Chooses a constrained slice subset of the RN table (start, stop, step)
        to give roughly num_mutations random numbers (less if overlap if
        step is too large)
        """

        return random_slices(rng, self._rand_num_table_size,
                             self._num_mutations, self._max_table_step)

    def _draw_random_parameter_slices(self, rng):
        """
        Chooses a constrained slice subset of the parameters (start, stop, step)
        to give roughly num_mutations perturbations (less if overlap if
        step is too large)
        """

        return random_slices(rng, self._num_parameters,
                             self._num_mutations, self._max_param_step)

    def _update(self, objective):

        # find slices in noise table and in parameters
        unmatched_dimension_slices = self._draw_random_parameter_slices(self._global_rng)
        unmatched_perturbation_slices = self._draw_random_table_slices(self._worker_rngs[self._rank])

        # Match slices against each other
        dimension_slices, perturbation_slices = match_slices(unmatched_dimension_slices,
                                                             unmatched_perturbation_slices)
        # Apply perturbations
        multi_slice_add(self._theta, self._rand_num_table, dimension_slices,
                        perturbation_slices, self._rank % 2 == 0)

        # Run objective
        local_rew = np.empty(1, dtype=np.float32)
        local_rew[0], roll_obs = objective(self._theta, curr_best=self._running_best_reward)

        if self.obj_kwargs['novelty']:
            behaviour = roll_obs[1]
            roll_obs = roll_obs[0]

        # Consolidate return values
        all_rewards = np.empty(self._size, dtype=np.float32)
        # Allgather is a blocking call, elements are gathered in order of rank
        self._comm.Allgather([local_rew, self._MPI.FLOAT],
                             [all_rewards, self._MPI.FLOAT])
        self._update_log(all_rewards)

        # save video of new best run judged by reward
        if self._rank == np.argmax(all_rewards) and np.max(all_rewards) >= self._running_best_reward and self._video:
            path = '{}-gen{}-rank{}-rew{:.2f}.mp4'.format(self._path, self._generation_number, self._rank, local_rew[0])
            save_video(roll_obs, path)
            print('## saved running best video to {}'.format(path))

        # treat behaviour/novelty:
        if self.obj_kwargs['novelty']:
            # put behaviour into archive with some probability (5/comm.size)
            size = 5 if self._comm_local.size >= 5 else self._size
            if self._comm_local.rank == 0:
                # choose ranks from which to pull novelty vectors
                ranks = np.random.RandomState(np.random.randint(10000)).choice(range(self._comm_local.size), size=size, replace=False)
            else:
                ranks = np.empty(size, dtype=np.int)
            self._comm_local.Bcast(ranks, root=0)
            # add chosen policy behaviours to archive
            for rank in ranks:
                if self._comm_local.rank == rank:
                    setoff = 0 if self._archive_idc == [] else self._archive_idc[-1][1]

                    # check if size of archive is depleted, in that case swap first entry
                    if setoff + len(behaviour) > len(self._archive):
                        # print("====== write at archive beginning again")
                        setoff = 0
                        self._archive_idc.pop(0)

                    bslice = [setoff, setoff + len(behaviour)]
                    self._archive_idc.append(bslice)
                    # clear space in archive for new behaviour if archive full
                    while True:
                        if self._archive_idc[0][0] >= bslice[1] or self._archive_idc[0][0] == 0:
                            break
                        else:
                            # print("====== threw out {}, new index {}".format(self._archive_idc[0], bslice))
                            self._archive_idc.pop(0)
                    self._archive[bslice[0]:bslice[1]] = behaviour

                self._archive_idc = self._comm_local.bcast(self._archive_idc, root=rank)

            # compute combined fitness
            local_novelty = np.empty(1, dtype=np.float32)
            local_novelty[0] = compute_novelty_vs_archive(self._archive, self._archive_idc, behaviour, size)
            all_novelty = np.empty(self._size, dtype=np.float32)
            self._comm.Allgather([local_novelty, self._MPI.FLOAT], [all_novelty, self._MPI.FLOAT])
            # rank rewards and novelty before calculating fitness, to get the same scales (Uber ES-NS Paper, sec.3.2)
            rew_ranks = np.empty_like(all_rewards)
            rew_ranks[np.argsort(all_rewards)] = np.arange(len(all_rewards))
            nov_ranks = np.empty_like(all_novelty)
            nov_ranks[np.argsort(all_novelty)] = np.arange(len(all_novelty))
            all_fitness = self._rew_nov_tradeoff * rew_ranks + (1 - self._rew_nov_tradeoff) * nov_ranks

        else:
            all_fitness = all_rewards

        self._comm.Barrier()
        # update theta according to fitness and log generation results
        if self._rank == 0:
            self._update_theta(all_fitness, unmatched_dimension_slices, dimension_slices, perturbation_slices, all_rewards if self.obj_kwargs['novelty'] else None)
            log(self, "Gen {} | Mean reward: {:.2f}, Best: {}, {:.2f} || Update ratio: {}".format(self._generation_number, np.sum(all_rewards) / self._size, np.argmax(all_rewards), np.max(all_rewards), self._update_ratios[-1]))
        else:
            self._update_theta(all_fitness, unmatched_dimension_slices, dimension_slices, perturbation_slices, all_rewards if self.obj_kwargs['novelty'] else None)

    def _update_theta(self, all_fitness, master_dim_slices, local_dim_slices,
                      local_perturbation_slices, all_rewards=None):
        """
        *Adds an additional multiply operation to avoid creating a new
        set of arrays for the slices. Not sure which would be faster
        *This is not applied to the running best, because it is not a weighted sum
        """

        if self._OpenAIES:
            g = np.zeros_like(self._theta, dtype=np.float32)
        else:
            g = self._old_theta

        if all_rewards is None:
            all_rewards = all_fitness

        best_rank = np.argmax(all_rewards)
        if all_rewards[best_rank] >= self._running_best_reward:
            self._update_best_flag = True
            self._running_best_reward = all_rewards[best_rank]
            log(self, "## New running best: {:.2f} ##".format(self._running_best_reward))
            self._running_best[:] = self._old_theta.copy()  # unperturbed theta

        for parent_num, rank in enumerate(np.argsort(all_fitness)[::-1]):
            if parent_num < self._num_parents:

                if rank == self._rank:
                    dimension_slices = local_dim_slices
                    perturbation_slices = local_perturbation_slices

                else:
                    perturbation_slices = self._draw_random_table_slices(self._worker_rngs[rank])
                    dimension_slices, perturbation_slices = match_slices(master_dim_slices, perturbation_slices)

                # Apply update running best
                # if parent_num == 0 and self._update_best_flag:
                if rank == best_rank and self._update_best_flag:
                    multi_slice_add(self._running_best, self._rand_num_table,
                                    dimension_slices, perturbation_slices, rank % 2 == 0)
                    self._update_best_flag = False
                    # print("updated running best, worker {}".format(self._rank), np.sum(self._running_best))

                # first divide the gradient with weight w, then add perturbation and multiply with w
                # again, such that only this perturbation is weighted in the end, supposedly faster
                multi_slice_divide(g, self._weights[parent_num], dimension_slices)
                multi_slice_add(g, self._rand_num_table, dimension_slices, perturbation_slices, rank % 2 == 0)
                multi_slice_multiply(g, self._weights[parent_num], dimension_slices)

            else:
                if rank != self._rank:
                    # advance the random draw for all other workers, such that they stay synchronised
                    perturbation_slices = self._draw_random_table_slices(self._worker_rngs[rank])
                    if rank == best_rank and self._update_best_flag:
                        dimension_slices, perturbation_slices = match_slices(master_dim_slices, perturbation_slices)
                        multi_slice_add(self._running_best, self._rand_num_table,
                                        dimension_slices, perturbation_slices, rank % 2 == 0)
                        self._update_best_flag = False

        # update step
        if self._OpenAIES:
            ur, self._old_theta = self.optimizer.update(g)
            self._update_ratios.append(ur)
            multi_slice_assign(self._theta, self._old_theta, master_dim_slices, master_dim_slices)
        else:  # old routine without optimizer, here g is already the new theta
            multi_slice_assign(self._theta, g, master_dim_slices, master_dim_slices)

    def _update_log(self, all_rewards):

        self._score_history.append(all_rewards)

    def _set_rew_nov_tradeoff(self, new_tradeoff):
        if self.obj_kwargs['novelty']:
            if new_tradeoff > 1.0:
                self._rew_nov_tradeoff = 1.0
                log(self, "====== keep on optimizing for REWARD, {:.1f}".format(self._rew_nov_tradeoff))
            elif new_tradeoff < 0.0:
                self._rew_nov_tradeoff = 0.0
                log(self, "====== keep on optimizing for NOVELTY, {:.1f}".format(self._rew_nov_tradeoff))
            else:
                if self._rew_nov_tradeoff > new_tradeoff:
                    log(self, "====== weighing stronger towards NOVELTY, {:.1f}".format(new_tradeoff))
                else:
                    log(self, "====== weighing stronger towards REWARD, {:.1f}".format(new_tradeoff))
                self._rew_nov_tradeoff = new_tradeoff

    def _set_sigma(self, new_sigma):
        if new_sigma < 0.00001:
            new_sigma = 0.00001
        elif new_sigma > 1.0:
            new_sigma = 1.0
        if self._sigma > new_sigma:
            log(self, "++++++ lowering SIGMA to, {:.5f}".format(new_sigma))
        else:
            log(self, "++++++ increasing SIGMA to, {:.5f}".format(new_sigma))
        self._weights *= np.array([self._sigma], dtype=np.float32)

        if self._comm_local.rank == 0:
            self._rand_num_table /= np.array([self._sigma], dtype=np.float32)
        self._sigma = new_sigma  # np.sqrt(10)
        self._weights /= np.array([self._sigma], dtype=np.float32)
        if self._comm_local.rank == 0:
            self._rand_num_table *= self._sigma

    @property
    def centroid(self):

        return self._theta.copy()

    @property
    def best(self):
        return self._running_best.copy()

    def showcase(self, save=False):
        print("Showcasing running best of: {:.2f}".format(self._running_best_reward))
        obs, rews = self.policy.play(self.env, theta=self._running_best.copy(), loop=True, save=save)
        if save:
            path = '{}.showcase.mp4'.format(self._path)
            save_video(obs, path)
            log(self, '## saved showcase to {}\n'.format(path))
            log(self, "Showcase performance: {:.2f} vs performance during Training: {:.2f}".format(np.mean(rews), self._running_best_reward))

    def plot_reward_over_time(self, prefix='test', logy=True, savefile=False):
        """
        Plots the evolutionary history of the population's reward.
        Includes min reward individual for each generation the mean
        """
        if self._rank == 0:
            import matplotlib.pyplot as plt

            rewards_by_generation = np.array(self._score_history)
            min_reward_by_generation = np.max(rewards_by_generation, axis=1)
            mean_reward_by_generation = np.mean(rewards_by_generation, axis=1)

            plt.plot(range(len(mean_reward_by_generation)),
                     mean_reward_by_generation,
                     marker='None', ls='-', color='blue', label='mean reward')

            plt.plot(range(len(min_reward_by_generation)),
                     min_reward_by_generation, ls='--', marker='None',
                     color='red', label='best')
            if logy:
                plt.yscale('log')
            plt.grid(True)
            plt.xlabel('generation')
            plt.ylabel('reward')
            plt.legend(loc='upper right')
            plt.tight_layout()

            if savefile:
                path = "save/" + prefix + "_evoreward.png"
                plt.savefig(path, dpi=300)
                plt.close()
                plt.clf()
                print("plotted to", path)
            else:
                plt.show()
                plt.clf()

    @classmethod
    def load(cls, filename):
        """
        Load a pickled file. Note that the objective function and any other
        non-picklable objects won't be saved.
        :param filename: name of pickled file
        :return: an evostrat object of this class
        """
        pickled_obj_file = open(filename, 'rb')
        # obj = pickle.load(pickled_obj_file)
        obj = torch.load(pickled_obj_file)
        pickled_obj_file.close()

        return obj

    def save(self, filename):
        """
        objectives and their args are not saved with the ES
        """
        if self._rank == 0:
            pickled_obj_file = open(filename, 'wb')
            # pickle.dump(self, pickled_obj_file, 2)
            torch.save(self, pickled_obj_file)
            pickled_obj_file.close()
            print("Saved to", filename)


def get_weight(recent_gens):
    # create symmetric weight
    if recent_gens % 2 == 0:
        w = list(range(-recent_gens // 2, 0)) + list(range(1, recent_gens // 2 + 1))
    else:
        w = list(range(-(recent_gens // 2), 0)) + [0] + list(range(1, recent_gens // 2 + 1))

    return (np.array(w) / (recent_gens // 2))**3
