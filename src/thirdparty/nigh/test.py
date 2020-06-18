import Nigh
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from sklearn.metrics import pairwise_distances

import pickle

w = float(5.0)
k=5
n = 15000
m = 300

seed = 5

tree = Nigh.SE3Tree(2 * w)
np.random.seed(seed)
t = np.random.normal(size=(n, 3)).astype('f')
q = np.random.normal(size=(n,4)).astype('f')
q = q / np.linalg.norm(q, axis=1)[:, np.newaxis]
R = Rotation.from_quat(q)
r = R.as_rotvec()
tree.insert(t, q)


t1 = np.random.normal(size=(m, 3)).astype('f')
q1 = np.random.normal(size=(m,4)).astype('f')
q1 = q1 / np.linalg.norm(q1, axis=1)[:, np.newaxis]
R1 = Rotation.from_quat(q1)
r1 = R1.as_rotvec()


start = time.time()
idx, dist = tree.nearest(t1, q1, k, 4)
print("time elapsed nighy ", time.time() - start)

stree = cKDTree(np.concatenate((t, w * r), axis=1))
start = time.time()
dist1, idx1 = stree.query(np.concatenate((t1, w * r1), axis=1), k, n_jobs=4)
print("time elapsed scipy ", time.time() - start)

#start = time.time()
dist_t = pairwise_distances(t1, t, n_jobs=4)
dist_R = np.arccos(np.abs(np.dot(q1, q.transpose())))

dist_T = dist_t +  2* dist_R * w
top_k_id = np.argsort(dist_T, axis=1)[:, :k]
top_k_dist = np.take_along_axis(dist_T, top_k_id, axis=1)
top_k_distR = np.take_along_axis(dist_R, top_k_id, axis=1)
top_k_distt = np.take_along_axis(dist_t, top_k_id, axis=1)
#print("time elapsed ", time.time() - start)
#print(top_k_id)
#print(top_k_dist)
#print(idx)
#print(dist)
#print(idx1)
#print(dist1)
#print(top_k_distt[2], top_k_distR[2])
#print(np.linalg.norm(r1[2] - r[2]))
#with open('file.pickle', 'wb') as f:
#    pickle.dump(tree, f)
