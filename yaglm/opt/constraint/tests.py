import numpy as np
from unittest import TestCase
from yaglm.opt.constraint import convex

class TestProjectionsOnConstraints(TestCase):
    def setUp(self):
        pass

    def assert_arrays_close(self, test_array, ref_array):
        "Custom assertion for arrays being almost equal"
        try:
            np.testing.assert_allclose(test_array, ref_array)
        except AssertionError:
            self.fail()

    def test_Positive(self):
        cons = convex.Positive()
        v = np.array([-1, 0, 2, 3, -2])
        self.assert_arrays_close(cons.prox(v), [0, 0, 2, 3, 0])
        self.assertEqual(cons.prox(-2), 0)

    def test_LinearEquality(self):
        A = np.identity(2)
        b = np.array([1,1])
        cons = convex.LinearEquality(A, b)
        proj = cons.prox(b)  # the proj of b should just be b
        self.assert_arrays_close(proj, b)

    def test_L2Ball(self):
        cons1 = convex.L2Ball(1)
        self.assert_arrays_close(cons1.prox([0,0,0]), [0,0,0])
        self.assert_arrays_close(cons1.prox([1,0,0]), [1,0,0])
        self.assert_arrays_close(cons1.prox([0.5,0,0]), [0.5,0,0])
        self.assert_arrays_close(cons1.prox([1,1,1]), np.array([1,1,1])/np.sqrt(3))
        self.assert_arrays_close(cons1.prox([1,-1,1]), np.array([1,-1,1])/np.sqrt(3))

        cons4 = convex.L2Ball(4)
        self.assert_arrays_close(cons4.prox([0,0,0]), [0,0,0])
        self.assert_arrays_close(cons4.prox([1,0,0]), [1,0,0])
        self.assert_arrays_close(cons4.prox([0.5,0,0]), [0.5,0,0])
        self.assert_arrays_close(cons4.prox([-4,3,0]), np.array([-4,3,0])/(5/4))

    def test_Isotonic(self):
        cons = convex.Isotonic(increasing=True)
        for v in [
                np.arange(5),
                np.array([-1, 0, 2, 3, -2]),
                np.array([-1, 3, 0, 3, 2])
            ]:
            result = cons.prox(v)
            lags = result[1:] - result[:-1]
            self.assertTrue((lags >= 0).all())

        cons = convex.Isotonic(increasing=False)
        for v in [
                np.arange(5),
                np.array([-1, 0, 2, 3, -2]),
                np.array([-1, 3, 0, 3, 2])
            ]:
            result = cons.prox(v)
            lags = result[1:] - result[:-1]
            self.assertTrue((lags <= 0).all())
