import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from src.geometry import SE3Poses, metric

np.random.seed(0)

class SE3PosesTests(unittest.TestCase):

    def setUp(self):
        self.n1 = 50
        self.n3 = 30
        self.w = 5

        self.t1 = np.random.normal(size=(self.n1, 3))
        self.R1 = Rotation(np.random.normal(size=(self.n1, 4)))
        self.t2 = np.random.normal(size=(self.n1, 3))
        self.R2 = Rotation(np.random.normal(size=(self.n1, 4)))
        self.t3 = np.random.normal(size=(self.n3, 3))
        self.R3 = Rotation(np.random.normal(size=(self.n3, 4)))
        self.t4 = np.random.normal(size=3)
        self.R4 = Rotation(np.random.normal(size=4))
        self.poses1 = SE3Poses(self.t1, self.R1)        
        self.poses2 = SE3Poses(self.t2, self.R2)        
        self.poses3 = SE3Poses(self.t3, self.R3)        
        self.poses4 = SE3Poses(self.t4, self.R4)        

    def test_init(self):
        # checks error is thrown if t and R have different sizes
        self.assertRaises(ValueError, SE3Poses, self.t1, self.R3)

    def test_slicing(self):
        # test slicing
        pose_slice = self.poses1[:5]
        np.testing.assert_almost_equal(pose_slice.t(), self.poses1.t()[:5])
        np.testing.assert_almost_equal((pose_slice.R().inv() * self.poses1.R()[:5]).magnitude(), np.zeros(5))
        # test single element
        pose_elem = self.poses1[0]
        np.testing.assert_almost_equal(pose_elem.t(), self.poses1.t()[0])
        np.testing.assert_almost_equal((pose_elem.R().inv() * self.poses1.R()[0]).magnitude(), 0.0)
    
    def test_len(self):
        self.assertEqual(len(self.poses1), self.n1)
        self.assertEqual(len(self.poses2), self.n1)
        self.assertEqual(len(self.poses3), self.n3)
        self.assertEqual(len(self.poses4), 1)

    def test_mul(self):
        poses12 = self.poses1 * self.poses2
        poses14 = self.poses1 * self.poses4
        poses41 = self.poses4 * self.poses1
        # compare operator * with manually applying transformations
        np.testing.assert_almost_equal(poses12.t(), self.poses1.R().apply(self.poses2.t()) + self.poses1.t())
        np.testing.assert_almost_equal((poses12.R().inv() * self.poses1.R() * self.poses2.R()).magnitude(), np.zeros(len(poses12)))
        # test broadcasting both directions
        for i, pose in enumerate(self.poses1):
            # left broadcast
            p41id = self.poses4 * pose / poses41[i]
            np.testing.assert_almost_equal(p41id.t(), np.zeros(3))
            self.assertAlmostEqual(p41id.R().magnitude(), 0.0)
            # right broadcast
            p14id = pose * self.poses4 / poses14[i]
            np.testing.assert_almost_equal(p14id.t(), np.zeros(3))
            self.assertAlmostEqual(p14id.R().magnitude(), 0.0)
        # test single 
        p410 = self.poses4 * self.poses1[0]
        p410id = p410.inv() * self.poses4 * self.poses1[0]
        np.testing.assert_almost_equal(p410id.t(), np.zeros(3))
        self.assertAlmostEqual(p410id.R().magnitude(), 0.0)


    def test_truediv(self):
        poses12 = self.poses1 / self.poses2
        poses14 = self.poses1 / self.poses4
        poses41 = self.poses4 / self.poses1
        # compare operator with direct computation
        np.testing.assert_almost_equal(self.poses1.R().inv().apply(self.poses2.t() - self.poses1.t()), poses12.t())
        np.testing.assert_almost_equal((poses12.R() * (self.poses2.R().inv() * self.poses1.R())).magnitude(), np.zeros(len(poses12)))
        # test broadcasting both directions
        for i, pose in enumerate(self.poses1):
            # left broadcast
            p41id = self.poses4 / pose * poses14[i]
            np.testing.assert_almost_equal(p41id.t(), np.zeros(3))
            self.assertAlmostEqual(p41id.R().magnitude(), 0.0)
            # right broadcast
            p14id = pose / self.poses4 * poses41[i]
            np.testing.assert_almost_equal(p14id.t(), np.zeros(3))
            self.assertAlmostEqual(p14id.R().magnitude(), 0.0)
        # test single 
        p410 = self.poses4 / self.poses1[0]
        p410id = self.poses4 * p410 * self.poses1[0].inv()
        np.testing.assert_almost_equal(p410id.t(), np.zeros(3))
        self.assertAlmostEqual(p410id.R().magnitude(), 0.0)

    def test_inv(self):
        identity1 = self.poses1.inv() * self.poses1
        identity2 = self.poses2.inv() * self.poses2
        identity3 = self.poses3.inv() * self.poses3
        # test translational identity (zero vector)
        np.testing.assert_almost_equal(identity1.t(), np.zeros_like(identity1.t()))
        np.testing.assert_almost_equal(identity2.t(), np.zeros_like(identity2.t()))
        np.testing.assert_almost_equal(identity3.t(), np.zeros_like(identity3.t()))
        # test rotational identity, note we use magnitude == 0 since quaternions are equivalent to multiple of +-1
        np.testing.assert_almost_equal(identity1.R().magnitude(), np.zeros(len(identity1)))
        np.testing.assert_almost_equal(identity2.R().magnitude(), np.zeros(len(identity2)))
        np.testing.assert_almost_equal(identity3.R().magnitude(), np.zeros(len(identity3)))

    def test_metric(self):
        self.assertRaises(ValueError, metric, self.poses1, self.poses3, self.w)
        self.assertRaises(ValueError, metric, self.poses1, self.poses2, -1)
        d12 = metric(self.poses1, self.poses2, self.w)
        d12_manual = np.sqrt(np.sum((self.poses1.t() - self.poses2.t()) ** 2, axis=1)) + self.w * (self.poses1.R().inv() * self.poses2.R()).magnitude()
        np.testing.assert_almost_equal(d12, d12_manual)
        # test broadcasting both directions
        dist14 = metric(self.poses1, self.poses4, self.w)
        dist41 = metric(self.poses4, self.poses1, self.w)
        for i, pose in enumerate(self.poses1):
            # left broadcast
            self.assertAlmostEqual(dist14[i], metric(pose, self.poses4, self.w))
            # right broadcast
            self.assertAlmostEqual(dist14[i], metric(self.poses4, pose, self.w))
        # single implicitly tested in above loop

if __name__ == '__main__':
    unittest.main()