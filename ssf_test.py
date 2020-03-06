import math
import numpy as np
import tracklib.filter as ft
import tracklib.utils as utils


def KFilter_test():
    N, T = 200, 1

    x_dim, z_dim = 4, 2
    qx, qy = math.sqrt(0.01), math.sqrt(0.02)
    rx, ry = math.sqrt(2), math.sqrt(1)

    # relevant matrix in state equation
    F = np.array([[1, 0, T, 0], [0, 1, 0, T], [0, 0, 1, 0], [0, 0, 0, 1]])
    Q = np.diag([qx**2, qy**2])
    L = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])

    # relevant matrix in measurement equation
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    R = np.diag([rx**2, ry**2])
    M = np.eye(*R.shape)

    # initial state and error convariance
    x = utils.col([1, 2, 0.2, 0.3])
    P = 100 * np.eye(x_dim)

    kf = ft.SSFilter(x_dim, z_dim, F, L, H, M, Q, R)
    kf.init(x, P)

    print(kf)

if __name__ == "__main__":
    KFilter_test()

    