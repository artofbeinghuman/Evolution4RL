import numpy as np


class Optimizer(object):
    def __init__(self, theta, l2coeff=0.005):
        self.theta = theta
        self.dim = len(self.theta)
        self.t = 0
        self.l2coeff = l2coeff

    def update(self, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        ratio = np.linalg.norm(step) / np.linalg.norm(self.theta)
        self.theta += step
        return ratio, self.theta

    def _compute_step(self, globalg):
        return globalg


class SGD(Optimizer):
    def __init__(self, theta, stepsize, l2coeff=0.005, momentum=0.9):
        Optimizer.__init__(self, theta)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.stepsize, self.momentum = stepsize, momentum

    def _compute_step(self, g):
        globalg = g - self.l2coeff * self.theta
        self.v = self.momentum * self.v + (1. - self.momentum) * globalg
        step = self.stepsize * self.v
        return step


class Adam(Optimizer):
    def __init__(self, theta, stepsize, l2coeff=0.005, beta1=0.9, beta2=0.999, epsilon=1e-08):
        Optimizer.__init__(self, theta)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, g):
        globalg = g - self.l2coeff * self.theta
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step
