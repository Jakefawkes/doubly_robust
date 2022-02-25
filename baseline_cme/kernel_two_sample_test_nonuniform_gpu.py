from __future__ import division
import numpy as np
from sys import stdout
from doubly_robust_method.kernels import *


def MMD2u(K, w, m, n):
    """The MMD^2_u unbiased statistic.
    """
    wx = 1.0 / w[:m]
    wy = 1.0 / (1.0 - w[m:])

    Kx = torch.outer(wx, wx) * K[:m, :m]
    Ky = torch.outer(wy, wy) * K[m:, m:]
    Kxy = torch.outer(wx, wy) * K[:m, m:]
    tot = 1.0 / (m * (m - 1.0)) * (Kx.sum() - torch.diag(Kx).sum()) + \
          1.0 / (n * (n - 1.0)) * (Ky.sum() - torch.diag(Ky).sum()) - \
          2.0 / (m * n) * Kxy.sum()
    return tot.item()


def compute_null_distribution(K, w, m, n, iterations=10000, verbose=False,
                              random_state=None, marker_interval=1000):
    """Compute the bootstrap null-distribution of MMD2u.
    """
    if type(random_state) == type(np.random.RandomState()):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    mmd2u_null = np.zeros(iterations)
    for i in range(iterations):
        if verbose and (i % marker_interval) == 0:
            print(i),
            stdout.flush()
        idx = rng.permutation(m + n)
        # K_i = K[idx, idx[:, None]]
        kernel_X = K[:, idx]
        kernel_X = kernel_X[idx, :]
        # w_i = w[idx]
        mmd2u_null[i] = MMD2u(kernel_X, w, m, n)

    if verbose:
        print("")

    return mmd2u_null


def compute_null_distribution_given_permutations(K, w, m, n, permutation,
                                                 iterations=None):
    """Compute the bootstrap null-distribution of MMD2u given
    predefined permutations.
    Note:: verbosity is removed to improve speed.
    """
    if iterations is None:
        iterations = len(permutation)

    mmd2u_null = np.zeros(iterations)
    for i in range(iterations):
        idx = permutation[i]
        K_i = K[idx, idx[:, None]]
        w_i = w[idx]
        mmd2u_null[i] = MMD2u(K_i, w_i, m, n)

    return mmd2u_null


def kernel_two_sample_test_nonuniform_gpu(X, Y, w, kernel_function='rbf', iterations=10000,
                                          verbose=False, random_state=None, ls=1.0):
    """Compute MMD^2_u, its null distribution and the p-value of the
    kernel two-sample test.
    Note that extra parameters captured by **kwargs will be passed to
    pairwise_kernels() as kernel parameters. E.g. if
    kernel_two_sample_test(..., kernel_function='rbf', gamma=0.1),
    then this will result in getting the kernel through
    kernel_function(metric='rbf', gamma=0.1).
    """
    m = len(X)
    n = len(Y)
    XY = torch.cat([X, Y], dim=0)

    # K = pairwise_kernels(XY, metric=kernel_function, **kwargs)

    if kernel_function == 'rbf':
        kernel = RBFKernel(XY)
        kernel._set_lengthscale(ls)
        K = kernel.evaluate()
    elif kernel_function == 'linear':
        kernel = LinearKernel(XY)
        K = kernel.evaluate()
    mmd2u = MMD2u(K, w, m, n)
    if verbose:
        print("MMD^2_u = %s" % mmd2u)
        print("Computing the null distribution.")

    mmd2u_null = compute_null_distribution(K, w, m, n, iterations,
                                           verbose=verbose,
                                           random_state=random_state)
    # p_value = max(1.0/iterations, (mmd2u_null > mmd2u).sum() /
    #              float(iterations))
    p_value = np.mean(mmd2u_null > mmd2u)

    if verbose:
        print("p-value ~= %s \t (resolution : %s)" % (p_value, 1.0 / iterations))

    return mmd2u, mmd2u_null, p_value

