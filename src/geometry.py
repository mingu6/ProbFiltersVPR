import numpy as np
import math
from numba import njit, prange

from scipy.spatial.transform import Rotation

# def invert_SE3(T):
#     """
#     Args:
#         T (np array Nx4x4): Array where each element is a 4x4 SE(3) transformation.
#     Returns:
#         T_inv (np array Nx4x4): corresponding inverse transformations of input as 
#                     4x4 SE(3) elements.
#     """
#     R = T[..., :3, :3]
#     t = T[..., :3, 3]

#     # compute R and t of inverse matrix
#     R_inv = R.transpose(0, 2, 1)
#     t_inv = np.zeros((len(T), 3))
#     t_inv = -R_inv @ t[:, :, np.newaxis]

#     lastrow = np.repeat(np.array([[0., 0., 0., 1.]]), len(T), axis=0)
#     # combine translation and rotation
#     T_inv = np.append(R_inv, t_inv, axis=2)
#     # add bottom row to make matrix 4x4 for SE(3) parameterization
#     T_inv = np.append(T_inv, lastrow[:, np.newaxis, :], axis=1)
#     return T_inv

class SE3Poses:
    def __init__(self, t, R):
        self._single = False

        if t.ndim not in [1, 2] or t.shape[-1] != 3:
            raise ValueError("Expected `t` to have shape (3,) or (N x 3), "
                             "got {}.".format(t.shape))

        # If a single quaternion is given, convert it to a 2D 1 x 4 matrix but
        # set self._single to True so that we can return appropriate objects
        # in the `to_...` methods
        if t.shape == (3,):
            t = t[None, :]
            self._single = True
            if len(R) > 1:            
                raise ValueError("Different number of translations 1 and rotations {}.".format(len(R)))
        elif len(t) == 1:
            self._single = True
        else:
            if len(t) != len(R):
                raise ValueError("Differing number of translations {} and rotations {}".format(len(t),len(R)))
        self._t = t
        self._R = R
        self.len = len(R)
    
    def __getitem__(self, indexer):
        return self.__class__(self.t()[indexer], self.R()[indexer])

    def __len__(self):
        return self.len

    def __mul__(self, other):
        """
        Performs element-wise pose composition.
        TO DO: Broadcasting
        """
        if not(len(self) == 1 or len(other) == 1 or len(self) == len(other)):
            raise ValueError("Expected equal number of transformations in both "
                             "or a single transformation in either object, "
                             "got {} transformations in first and {} transformations in "
                             "second object.".format(
                                len(self), len(other)))
        return self.__class__(self.R().apply(other.t()) + self.t(), self.R() * other.R())

    def __truediv__(self, other):
        """
        Computes relative pose, similar to MATLAB convention (x = A \ b for Ax = b). Example:
        T1 / T2 = T1.inv() * T2
        TO DO: Broadcasting
        """
        if not(len(self) == 1 or len(other) == 1 or len(self) == len(other)):
            raise ValueError("Expected equal number of transformations in both "
                             "or a single transformation in either object, "
                             "got {} transformations in first and {} transformations in "
                             "second object.".format(
                                len(self), len(other)))
        R1_inv = self.R().inv()
        t_new = R1_inv.apply(other.t() - self.t())
        return self.__class__(t_new, R1_inv * other.R())

    def t(self):
        return self._t[0] if self._single else self._t

    def R(self):
        return self._R

    def inv(self):
        R_inv = self.R().inv()
        t_new = -R_inv.apply(self.t())
        return SE3Poses(t_new, R_inv)

    def components(self):
        """
        Return translational and rotational components of pose separately. Quaternion form for rotations.
        """
        return self.t(), self.R()

    def repeat(self, N):
        t = self.t()
        q = self.R().as_quat()
        if len(self) == 1:
            t = np.expand_dims(t, 0)
            q = np.expand_dims(q, 0)
        t = np.repeat(t, N, axis=0)
        q = np.repeat(q, N, axis=0)
        return SE3Poses(t, Rotation.from_quat(q))

def metric(p1, p2, w):
    """
    Computes metric on the cartesian product space representation of SE(3).
    Args:
        p1 (SE3Poses) : set of poses
        p2 (SE3Poses) : set of poses (same size as p1)
        w (float > 0) : weight for attitude component
    """
    if not(len(p1) == 1 or len(p2) == 1 or len(p1) == len(p2)):
        raise ValueError("Expected equal number of transformations in both "
                            "or a single transformation in either object, "
                            "got {} transformations in first and {} transformations in "
                            "second object.".format(
                            len(p1), len(p2)))
    if w < 0:
        raise ValueError("Weight must be non-negative, currently {}".format(w))
    p_rel = p1 / p2
    t_dist = np.linalg.norm(p_rel.t(), axis=-1)
    R_dist = p_rel.R().magnitude()
    return t_dist + w * R_dist 

def combine(listOfPoses):
    """
    combines a list of SE3 objects into a single SE3 object
    """
    tList = []
    qList = []
    for pose in listOfPoses:
        if len(pose) == 1:
            t_temp = np.expand_dims(pose.t(), 0)
            q_temp = np.expand_dims(pose.R().as_quat(), 0)
        else:
            t_temp = pose.t()
            q_temp = pose.R().as_quat()
        tList.append(t_temp)
        qList.append(q_temp)
    tList = np.concatenate(tList, axis=0)
    qList = np.concatenate(qList, axis=0)
    return SE3Poses(tList, Rotation.from_quat(qList))

def expSE3(twist):
    """
    Applies exponential map to twist vectors
    Args
        twist: N x 6 matrix or 6D vector containing se(3) element(s)
    Returns
        SE3Poses object with equivalent SE3 transforms
    """
    u = twist[..., :3]
    w = twist[..., 3:]

    R = Rotation.from_rotvec(w)
    theta = R.magnitude()
    what = hatOp(w) # skew symmetric form
    with np.errstate(divide='ignore'):
        B = (1 - np.cos(theta)) / theta ** 2
        C = (theta - np.sin(theta)) / theta ** 3
    if len(twist.shape) == 2:
        B[np.abs(theta) < 1e-3] = 0.5 # limit for theta -> 0
        C[np.abs(theta) < 1e-3] = 1. / 6 # limit for theta -> 0
        V = np.eye(3)[np.newaxis, ...] + B[:, np.newaxis, np.newaxis] * what + C[:, np.newaxis, np.newaxis] * what @ what
        V = V.squeeze()
    else:
        if np.abs(theta) < 1e-3:
            B = 0.5 # limit for theta -> 0
            C = 1. / 6 # limit for theta -> 0
        V = np.eye(3) + B * what + C * what @ what
    t = V @ u[..., np.newaxis]
    return SE3Poses(t.squeeze(), R)

def logSE3(T):
    """
    Applies inverse exponential map to SE3 elements
    Args
        T: SE3Poses element, may have 1 or more (N) poses
    Returns
        Nx6 matrix or 6D vector representing twists
    """
    R = T.R() 
    t = T.t()

    theta = R.magnitude()
    with np.errstate(divide='ignore', invalid='ignore'):
        A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / theta ** 2
        sq_coeff = 1 / theta ** 2 * (1 - A / (2 * B))
    what = hatOp(R.as_rotvec())

    if len(T) > 1:
        A[np.abs(theta) < 1e-3] = 1. # limit for theta -> 0
        B[np.abs(theta) < 1e-3] = 0.5 # limit for theta -> 0
        sq_coeff[np.abs(theta) < 1e-3] = 1. / 12
        Vinv = np.eye(3)[np.newaxis, ...] - 0.5 * what + sq_coeff[:, np.newaxis, np.newaxis] * what @ what
    else:
        if np.abs(theta) < 1e-3:
            A = 1. # limit for theta -> 0
            B = 0.5 # limit for theta -> 0
            sq_coeff = 1. / 12
        Vinv = np.eye(3) - 0.5 * what + sq_coeff * what @ what
    u = Vinv @ t[..., np.newaxis]
    return np.concatenate((u.squeeze(), R.as_rotvec()), axis=-1)

def hatOp(vec):
    """
    Turns Nx3 vector into Nx3x3 skew skymmetric representation. Works for single 3D vector also.
    """
    if len(vec.shape) == 2:
        mat = np.zeros((vec.shape[0], 3, 3))
    else:
        mat = np.zeros((3, 3))
    mat[..., 1, 0] = vec[..., 2]
    mat[..., 2, 1] = vec[..., 0]
    mat[..., 0, 2] = vec[..., 1]
    if len(vec.shape) == 2:
        mat = mat - np.transpose(mat, (0, 2, 1))
    else:
        mat = mat - mat.transpose()
    return mat

def vOp(mat):
    """
    Turns Nx3x3 skew symmetric matrices to Nx3 vectors. Works for a single 3x3 matrix also.
    """
    if len(mat.shape) == 3:
        vec = np.zeros((vec.shape[0], 3))
    else:
        vec = np.zeros(3)
    vec[..., 0] = mat[..., 2, 1]
    vec[..., 1] = mat[..., 0, 2]
    vec[..., 2] = mat[..., 1, 0]
    return vec

@njit(parallel=True, cache=True)
def fast_dot(A1, A2):
    """
    For two m x n matrices, compute dot products between each of the rows.
    Returns length m vector consisting of row-wise dot products.

    """
    M = A1.shape[0]
    D = A1.shape[1]
    dots = np.empty(M)
    for i in prange(M):
        dot = 0.0
        for j in range(D):
            dot += A1[i, j] * A2[i, j]
        dots[i] = dot
    return dots

@njit(parallel=True, cache=True)
def fast_distance(A1, A2):
    M = A1.shape[0]
    D = A1.shape[1]
    dists = np.empty(M)
    for i in prange(M):
        sum_sq = 0.
        for j in range(D):
            tmp = (A1[i, j] - A2[i, j]) 
            sum_sq += tmp * tmp
        dists[i] = np.sqrt(sum_sq)
    return dists

@njit(parallel=True, cache=True)
def fast_distance_pairwise(A1, A2):
    dists = np.empty((len(A1), len(A2)))
    M = A1.shape[0]
    N = A2.shape[0]
    D = A1.shape[1]
    for i in prange(len(A1)):
        for j in range(len(A2)):
            # dists[i, j] = np.linalg.norm(A1[i] - A2[j])
            sum_sq = 0.
            for k in range(D):
                tmp = (A1[i, k] - A2[j, k]) 
                sum_sq += tmp * tmp
            dists[i, j] = np.sqrt(sum_sq)
    return dists