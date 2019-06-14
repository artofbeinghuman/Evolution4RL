import numpy as np
from functools import partial
import pickle
import torch
from anfang import Policy
from optimizers import SGD, Adam


def multi_slice_add(x1_inplace, x2, x1_slices=(), x2_slices=(), add=True):
    """
    Does an inplace addition (subtraction for add=False) on x1 given a list
    of slice objects. If slices for both are given, it is assumed that they 
    will be of the same size and each slice will have the same # of elements
    """

    if add:
        if (len(x1_slices) != 0) and (len(x2_slices) == 0):
            for x1_slice in x1_slices:
                x1_inplace[x1_slice] += x2

        elif (len(x1_slices) != 0) and (len(x2_slices) != 0) \
                and (len(x2_slices) == len(x1_slices)):

            for i in range(len(x1_slices)):
                x1_inplace[x1_slices[i]] += x2[x2_slices[i]]

        elif (len(x1_slices) == 0) and (len(x2_slices) == 0):
            x1_inplace += x2
    else:
        if (len(x1_slices) != 0) and (len(x2_slices) == 0):
            for x1_slice in x1_slices:
                x1_inplace[x1_slice] -= x2

        elif (len(x1_slices) != 0) and (len(x2_slices) != 0) \
                and (len(x2_slices) == len(x1_slices)):

            for i in range(len(x1_slices)):
                x1_inplace[x1_slices[i]] -= x2[x2_slices[i]]

        elif (len(x1_slices) == 0) and (len(x2_slices) == 0):
            x1_inplace -= x2


def multi_slice_multiply(x1_inplace, x2, x1_slices=(), x2_slices=()):
    """
    Does an inplace multiplication on x1 given a list of slice objects
    If slices for both are given, it is assumed that they will be of
    the same size and each slice will have the same # of elements
    """

    if (len(x1_slices) != 0) and (len(x2_slices) == 0):
        for x1_slice in x1_slices:
            x1_inplace[x1_slice] *= x2

    elif (len(x1_slices) != 0) and (len(x2_slices) != 0) \
            and (len(x2_slices) == len(x1_slices)):

        for i in range(len(x1_slices)):
            x1_inplace[x1_slices[i]] *= x2[x2_slices[i]]

    elif (len(x1_slices) == 0) and (len(x2_slices) == 0):
        x1_inplace *= x2


def multi_slice_divide(x1_inplace, x2, x1_slices=(), x2_slices=()):
    """
    Does an inplace multiplication on x1 given a list of slice objects
    If slices for both are given, it is assumed that they will be of
    the same size and each slice will have the same # of elements
    """

    if (len(x1_slices) != 0) and (len(x2_slices) == 0):
        for x1_slice in x1_slices:
            x1_inplace[x1_slice] /= x2

    elif (len(x1_slices) != 0) and (len(x2_slices) != 0) \
            and (len(x2_slices) == len(x1_slices)):

        for i in range(len(x1_slices)):
            x1_inplace[x1_slices[i]] /= x2[x2_slices[i]]

    elif (len(x1_slices) == 0) and (len(x2_slices) == 0):
        x1_inplace /= x2


def multi_slice_assign(x1_inplace, x2, x1_slices=(), x2_slices=()):
    """
    Does an inplace assignment on x1 given a list of slice objects
    If slices for both are given, it is assumed that they will be of
    the same size and each slice will have the same # of elements
    """

    if (len(x1_slices) != 0) and (len(x2_slices) == 0):
        for x1_slice in x1_slices:
            x1_inplace[x1_slice] = x2

    elif (len(x1_slices) != 0) and (len(x2_slices) != 0) \
            and (len(x2_slices) == len(x1_slices)):

        for i in range(len(x1_slices)):
            x1_inplace[x1_slices[i]] = x2[x2_slices[i]]

    elif (len(x1_slices) == 0) and (len(x2_slices) == 0):
        x1_inplace = x2


def multi_slice_mod(x1_inplace, x2, x1_slices=(), x2_slices=()):
    """
    Does an inplace modulo on x1 given a list of slice objects
    If slices for both are given, it is assumed that they will be of
    the same size and each slice will have the same # of elements
    """

    if (len(x1_slices) != 0) and (len(x2_slices) == 0):
        for x1_slice in x1_slices:
            x1_inplace[x1_slice] %= x2

    elif (len(x1_slices) != 0) and (len(x2_slices) != 0) \
            and (len(x2_slices) == len(x1_slices)):

        for i in range(len(x1_slices)):
            x1_inplace[x1_slices[i]] %= x2[x2_slices[i]]

    elif (len(x1_slices) == 0) and (len(x2_slices) == 0):
        x1_inplace %= x2


def multi_slice_fabs(x1_inplace, x1_slices=()):
    """
    Does an inplace fabs on x1 given a list of slice objects
    """

    if len(x1_slices) != 0:
        for x1_slice in x1_slices:
            np.fabs(x1_inplace[x1_slice], out=x1_inplace[x1_slice])

    else:
        np.fabs(x1_inplace, out=x1_inplace)


def multi_slice_clip(x1_inplace, lower, upper, xslices=None,
                     lslices=None, uslices=None):
    """
    Does an inplace clip on x1
    """

    if (lslices is None) and (uslices is None) and (xslices is None):
        np.clip(x1_inplace, lower, upper, out=x1_inplace)

    elif (lslices is None) or (uslices is None) and (xslices is not None):
        for xslice in xslices:
            np.clip(x1_inplace[xslice], lower, upper, out=x1_inplace[xslice])

    elif (lslices is not None) and (uslices is not None) \
            and (len(lslices) == len(uslices) and (xslices is not None)):
        for i in range(len(xslices)):
            np.clip(x1_inplace[xslices[i]], lower[lslices[i]], upper[uslices[i]],
                    out=x1_inplace[xslices[i]])

    else:
        raise NotImplementedError("Invalid arguments in multi_slice_clip")


def random_slices(rng, iterator_size, size_of_slice, max_step=1):
    """
    Returns a list of slice objects given the size of the iterator it
    will be used for and the number of elements desired for the slice
    This will return additional slice each time it wraps around the
    iterator
    iterator_size - the number of elements in the iterator
    size_of_slice - the number of elements the slices will cover
    max_step - the maximum number of steps a slice will take.
                This affects the number of slice objects created, as
                larger max_step will create more wraps around the iterator
                and so return more slice objects
    The number of elements is not guaranteed when slices overlap themselves
    """

    step_size = rng.randint(1, max_step + 1)  # randint is exclusive
    start_step = rng.randint(0, iterator_size)

    return build_slices(start_step, iterator_size, size_of_slice, step_size)


def build_slices(start_step, iterator_size, size_of_slice, step_size):
    """
    Given a starting index, the size of the total members of the window,
    a step size, and the size of the iterator the slice will act upon,
    this function returns a list of slice objects that will cover that full
    window. Upon reaching the endpoints of the iterator, it will wrap around.
    """

    if step_size >= iterator_size:
        raise NotImplementedError("Error: step size must be less than the "
                                  + "size of the iterator")
    end_step = start_step + step_size * size_of_slice
    slices = []
    slice_start = start_step
    for i in range(1 + (end_step - step_size) // iterator_size):
        remaining = end_step - i * iterator_size
        if remaining > iterator_size:
            remaining = iterator_size

        slice_end = (slice_start + 1) + ((remaining
                                          - (slice_start + 1)) // step_size) * step_size
        slices.append(np.s_[slice_start:slice_end:step_size])  # s_ creates slice
        slice_start = (slice_end - 1 + step_size) % iterator_size

    return slices


def match_slices(slice_list1, slice_list2):
    """
    Will attempt to create additional slices to match the # elements of
    each slice from list1 to the corresponding slice of list 2.
    Will fail if the total # elements is different for each list
    """

    slice_list1 = list(slice_list1)
    slice_list2 = list(slice_list2)
    if slice_size(slice_list1) == slice_size(slice_list2):
        slice_list1.reverse()  # reverse list, to pop starting from 0 down to -1
        slice_list2.reverse()
        new_list1_slices = []
        new_list2_slices = []

        while len(slice_list1) != 0 and len(slice_list2) != 0:
            slice_1 = slice_list1.pop()
            slice_2 = slice_list2.pop()
            size_1 = slice_size(slice_1)
            size_2 = slice_size(slice_2)

            if size_1 < size_2:
                new_slice_2, slice_2 = splice_slice(slice_2, size_1)
                slice_list2.append(slice_2)
                new_list2_slices.append(new_slice_2)
                new_list1_slices.append(slice_1)

            elif size_2 < size_1:
                new_slice_1, slice_1 = splice_slice(slice_1, size_2)
                slice_list1.append(slice_1)
                new_list1_slices.append(new_slice_1)
                new_list2_slices.append(slice_2)

            elif size_1 == size_2:
                new_list1_slices.append(slice_1)
                new_list2_slices.append(slice_2)

    else:
        raise AssertionError("Error: slices not compatible")

    return new_list1_slices, new_list2_slices


def splice_slice(slice_obj, num_elements):
    """
    Returns two slices spliced from a single slice.
    The size of the first slice will be # elements
    The size of the second slice will be the remainder
    """

    splice_point = slice_obj.step * (num_elements - 1) + slice_obj.start + 1
    new_start = splice_point - 1 + slice_obj.step
    return np.s_[slice_obj.start: splice_point: slice_obj.step], \
        np.s_[new_start: slice_obj.stop: slice_obj.step]


def slice_size(slice_objects):
    """
    Returns the total number of elements in the combined slices
    Also works if given a single slice
    """

    num_elements = 0

    try:
        for sl in slice_objects:
            num_elements += (sl.stop - (sl.start + 1)) // sl.step + 1
    except TypeError:
        num_elements += (slice_objects.stop - (slice_objects.start + 1)) \
            // slice_objects.step + 1

    return num_elements


def get_env_from(config):
    import gym
    env = gym.make(config['env_id'])
    if config['policy']['type'] == "ESAtariPolicy":
        from .atari_wrappers import wrap_deepmind
        env = wrap_deepmind(env)
    return env


# TODO: CHANGE COST TO REWARDS, THUS CHANGE LOGIC TO MAXIMIZE REWARDS INSTEAD OF MINIMIZE COST
# TODO: implement mirrored sampling

class ES:
    """
    Runs a basic distributed Evolutionary strategy using MPI. The centroid
    is updated via a log-rank weighted sum of the population members that are
    generated from a normal distribution. This class evaluates a COST function
    rather than a fitness function.

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

    def __init__(self, config, xo, step_size, **kwargs):
        """
        :param xo: initial centroid
        :param step_size: float. the size of mutations
        :param num_mutations: number of dims to mutate in each iter (defaults to #dim)
        :param verbose: True/False. print info on run
        """

        # Initiate MPI
        from mpi4py import MPI
        self._MPI = MPI
        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()
        assert self._size % 2 == 0

        self._global_seed = kwargs.get('seed', 123)
        torch.manual_seed(self._global_seed)

        # create policy of this worker
        # (parameters should be initialised randomly via global rng, such that all workers start with identical centroid)
        self.config = config
        self.env = get_env_from(config)
        self.policy = Policy(self.env)
        self._theta = self.policy.get_flat()
        self.optimizer = {'sgd': SGD, 'adam': Adam}[config['optimizer']['type']](self._theta, **config['optimizer']['args'])

        # User input parameters
        self.objective = kwargs.get('objective', None)
        self.obj_kwargs = kwargs.get('obj_kwargs', {})
        self._step_size = np.float32(step_size)
        self._num_parameters = len(xo)
        self._num_mutations = kwargs.get('num_mutations', self._num_parameters)  # limited perturbartion ES as in Zhang et al 2017, ch.4.1
        self._verbose = kwargs.get('verbose', False)

        self._OpenAIES = kwargs.get('OpenAIES', True)

        if self._OpenAIES:
            # use centered ranks [0.5, -0.5]
            self._num_parents = self._size
            self._weights = np.arange(self._num_parents, 0, -1)
            self._weights /= self._num_parents
            self._weights -= 0.5
        else:
            # Classics ES:
            self._num_parents = kwargs.get('num_parents', int(self._size / 2))  # truncated selection
            self._weights = np.log(self._num_parents + 0.5) - np.log(
                np.arange(1, self._num_parents + 1))
            self._weights /= np.sum(self._weights)
            self._weights.astype(np.float32, copy=False)

        # Internal data
        self._global_rng = np.random.RandomState(self._global_seed)
        self._seed_set = self._global_rng.choice(1000000, size=self._size // 2,
                                                 replace=False)
        # for mirrored sampling, two subsequent workers (according to rank) will have the same seed/rng
        # the even ranked workers will perturb positively, the odd ones negatively
        self._seed_set = [s for ss in [[s, s] for s in self._seed_set] for s in ss]
        self._worker_rngs = [np.random.RandomState(seed) for seed in self._seed_set]
        self._par_choices = np.arange(0, self._num_parameters, dtype=np.int32)
        self._generation_number = 0
        self._score_history = []

        # State
        self._old_theta = self._theta.copy()
        self._running_best = self._theta.copy()
        self._running_best_cost = np.float32(np.finfo(np.float32).max)
        self._update_best_flag = False

        # int(5 * 10**8)
        self._rand_num_table_size = kwargs.get("rand_num_table_size", 250000000)  # 1GB of noise
        nbytes = self._rand_num_table_size * MPI.FLOAT.Get_size() if self._rank == 0 else 0
        win = MPI.Win.Allocate_shared(nbytes, MPI.FLOAT.Get_size(), comm=self._comm)
        buf, itemsize = win.Shared_query(0)
        assert itemsize == MPI.FLOAT.Get_size()
        self._rand_num_table = np.ndarray(buffer=buf, dtype='f', shape=(self._rand_num_table_size,))
        if self._rank == 0:
            self._rand_num_table[:] = self._global_rng.randn(self._rand_num_table_size)
            # Fold step-size into table values
            self._rand_num_table *= self._step_size

        self._max_table_step = kwargs.get("max_table_step", 5)
        self._max_param_step = kwargs.get("max_param_step", 1)

        self._comm.Barrier()

    def __getstate__(self):

        state = {"_step_size": self._step_size,
                 "_num_parameters": self._num_parameters,
                 "_num_mutations": self._num_mutations,
                 "_num_parents": self._num_parents,
                 "_verbose": self._verbose,
                 "_global_seed": self._global_seed,
                 "_weights": self._weights,
                 "_global_rng": self._global_rng,
                 "_seed_set": self._seed_set,
                 "_worker_rngs": self._worker_rngs,
                 "_generation_number": self._generation_number,
                 "_score_history": self._score_history,
                 "_theta": self._theta,
                 "_running_best": self._running_best,
                 "_running_best_cost": self._running_best_cost,
                 "_update_best_flag": self._update_best_flag,
                 "_rand_num_table_size": self._rand_num_table_size,
                 "_max_table_step": self._max_table_step,
                 "_max_param_step": self._max_param_step}

        return state

    def __setstate__(self, state):

        for key in state:
            setattr(self, key, state[key])

        # Reconstruct larger structures and load MPI
        self._par_choices = np.arange(0, self._num_parameters, dtype=np.int32)
        self._old_theta = self._theta.copy()
        from mpi4py import MPI
        self._MPI = MPI
        self._comm = MPI.COMM_WORLD
        self._size = self._comm.Get_size()
        self._rank = self._comm.Get_rank()
        self.objective = None
        self.obj_args = None

        self._rand_num_table = self._global_rng.randn(self._rand_num_table_size)
        self._rand_num_table *= self._step_size

    def __call__(self, num_iterations, objective=None, kwargs=None):
        """
        :param num_iterations: how many generations it will run for
        :param objective: a full or partial version of function
        :param kwargs: key word arguments for additional objective parameters
        :return: None
        """
        if objective is not None:
            if kwargs is None:
                kwargs = {}
        elif (self.objective is None) and (objective is None):
            raise AttributeError("Error: No objective defined")
        else:
            objective = self.objective
            kwargs = self.obj_kwargs

        partial_objective = partial(objective, **kwargs)
        for i in range(num_iterations):
            if self._verbose and (self._rank == 0):
                print("Generation:", self._generation_number)
            self._update(partial_objective)
            self._generation_number += 1

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

        # Perturb centroid
        unmatched_dimension_slices = self._draw_random_parameter_slices(
            self._global_rng)
        unmatched_perturbation_slices = self._draw_random_table_slices(
            self._worker_rngs[self._rank])

        # Match slices against each other
        dimension_slices, perturbation_slices = match_slices(
            unmatched_dimension_slices,
            unmatched_perturbation_slices)
        # Apply perturbations
        multi_slice_add(self._theta, self._rand_num_table,
                        dimension_slices, perturbation_slices, self._rank % 2 == 0)

        # Run objective
        local_cost = np.empty(1, dtype=np.float32)
        local_cost[0] = objective(self._theta)

        # Consolidate return values
        all_costs = np.empty(self._size, dtype=np.float32)
        # Allgather is a blocking call, elements are gathered in order of rank
        self._comm.Allgather([local_cost, self._MPI.FLOAT],
                             [all_costs, self._MPI.FLOAT])
        self._update_log(all_costs)

        self._update_theta(all_costs, unmatched_dimension_slices,
                           dimension_slices, perturbation_slices)

    def _reconstruct_perturbation(self, rank, master_dim_slices, parent_num):
        """
        Constructs the perturbation from other nodes given their rank (from
        which the seed is determined). Also updates the old_theta with
        the weighted perturbation
        :param rank: the rank of the node
        :param master_dim_slices: the indices that everyone is mutating
        :param parent_num: the member performance
        :return: perturbation_slices
        """
        perturbation_slices = self._draw_random_table_slices(
            self._worker_rngs[rank])
        dimension_slices, perturbation_slices = match_slices(
            master_dim_slices,
            perturbation_slices)
        multi_slice_divide(self._old_theta, self._weights[parent_num],
                           dimension_slices)
        multi_slice_add(self._old_theta, self._rand_num_table,
                        dimension_slices, perturbation_slices, rank % 2 == 0)
        multi_slice_multiply(self._old_theta, self._weights[parent_num],
                             dimension_slices)

        return perturbation_slices, dimension_slices

    def _update_theta(self, all_costs, master_dim_slices, local_dim_slices,
                      local_perturbation_slices):
        """
        *Adds an additional multiply operation to avoid creating a new
        set of arrays for the slices. Not sure which would be faster
        *This is not applied to the running best, because it is not a weighted sum
        """
        for parent_num, rank in enumerate(np.argsort(all_costs)):
            if parent_num < self._num_parents:
                # Begin Update sequence for the running best if applicable
                if (parent_num == 0) and (all_costs[rank] <= self._running_best_cost):
                    self._update_best_flag = True
                    self._running_best_cost = all_costs[rank]
                    # Since the best is always first, copy centroid elements
                    self._running_best[:] = self._old_theta

                if rank == self._rank:

                    # Update best for self ranks
                    if (parent_num == 0) and self._update_best_flag:
                        multi_slice_add(self._running_best, self._rand_num_table,
                                        local_dim_slices, local_perturbation_slices, rank % 2 == 0)

                    multi_slice_divide(self._old_theta, self._weights[parent_num],
                                       local_dim_slices)
                    multi_slice_add(self._old_theta, self._rand_num_table,
                                    local_dim_slices, local_perturbation_slices, rank % 2 == 0)
                    multi_slice_multiply(self._old_theta, self._weights[parent_num],
                                         local_dim_slices)

                else:
                    perturbation_slices, dimension_slices = \
                        self._reconstruct_perturbation(rank, master_dim_slices,
                                                       parent_num)

                    # Apply update best for non-self ranks
                    if (parent_num == 0) and self._update_best_flag:
                        multi_slice_add(self._running_best, self._rand_num_table,
                                        dimension_slices, perturbation_slices, rank % 2 == 0)

            else:
                if rank != self._rank:
                    self._draw_random_table_slices(self._worker_rngs[rank])

        multi_slice_assign(self._theta, self._old_theta,
                           master_dim_slices, master_dim_slices)

    def _update_log(self, costs):

        self._score_history.append(costs)

    @property
    def centroid(self):

        return self._theta.copy()

    @property
    def best(self):

        return self._running_best.copy()

    def plot_cost_over_time(self, prefix='test', logy=True, savefile=False):
        """
        Plots the evolutionary history of the population's cost.
        Includes min cost individual for each generation the mean
        """
        if self._rank == 0:
            import matplotlib.pyplot as plt

            costs_by_generation = np.array(self._score_history)
            min_cost_by_generation = np.min(costs_by_generation, axis=1)
            mean_cost_by_generation = np.mean(costs_by_generation, axis=1)

            plt.plot(range(len(mean_cost_by_generation)),
                     mean_cost_by_generation,
                     marker='None', ls='-', color='blue', label='mean cost')

            plt.plot(range(len(min_cost_by_generation)),
                     min_cost_by_generation, ls='--', marker='None',
                     color='red', label='best')
            if logy:
                plt.yscale('log')
            plt.grid(True)
            plt.xlabel('generation')
            plt.ylabel('cost')
            plt.legend(loc='upper right')
            plt.tight_layout()

            if savefile:
                plt.savefig(prefix + "_evocost.png", dpi=300)
                plt.close()
                plt.clf()
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
        obj = pickle.load(pickled_obj_file)
        pickled_obj_file.close()

        return obj

    def save(self, filename):
        """
        objectives and their args are not saved with the ES
        """
        if self._rank == 0:
            pickled_obj_file = open(filename, 'wb')
            pickle.dump(self, pickled_obj_file, 2)
            pickled_obj_file.close()
