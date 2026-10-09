import unittest

import numpy as np

from srdstiff.core.types import SRDStiffConfig
from srdstiff.solvers.srd_stiff import SRDStiffSolver


class RosenbrockObjective:
    """Mock Rosenbrock problem for testing."""
    name = "ROSENBR_MOCK"
    n = 2

    @property
    def x_init(self):
        return np.array([-1.2, 1.0])

    def f(self, x):
        return (1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2

    def grad(self, x):
        g = np.zeros(2)
        g[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0]**2)
        g[1] = 200.0 * (x[1] - x[0]**2)
        return g

    def hess(self, x):
        h = np.zeros((2, 2))
        h[0, 0] = 2.0 - 400.0 * (x[1] - 3.0 * x[0]**2)
        h[0, 1] = -400.0 * x[0]
        h[1, 0] = -400.0 * x[0]
        h[1, 1] = 200.0
        return h

    def hess_vec(self, x, v):
        return self.hess(x) @ v

    def hvp(self, x, v):
        return self.hess_vec(x, v)


class TestSRDStiffSolver(unittest.TestCase):

    def test_rosenbrock_convergence(self):
        obj = RosenbrockObjective()
        config = SRDStiffConfig(
            seed=42,
            max_iter=1000,
            alpha_init=1.0,
            K=500.0,
            gamma=0.1,
            epsilon=0.1,
            delta=0.05,
            L_H_estimate=1.0,
            tol_grad_abs=1e-5,
            tol_grad_rel=1e-7,
        )
        solver = SRDStiffSolver(config)
        result = solver.solve(obj)
        
        self.assertTrue(result.is_converged())
        self.assertTrue(result.grad_norm_final <= 1e-4)
        
        # Orijinal noktadan uzakta mıyız kontrolü ve çözüme (1, 1) yakın mıyız
        self.assertTrue(np.allclose(result.x_final, [1.0, 1.0], atol=1e-3))


if __name__ == "__main__":
    unittest.main()
