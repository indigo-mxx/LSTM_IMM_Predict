"""IMM 核心行为测试。"""

import unittest

import numpy as np

from src.imm import IMMFilter


class IMMFilterTest(unittest.TestCase):
    def test_匀速轨迹能够降低位置误差(self) -> None:
        rng = np.random.default_rng(7)
        truth = np.column_stack([np.arange(40, dtype=float), np.arange(40, dtype=float) * 0.4])
        measurements = truth + rng.normal(0.0, 0.9, size=truth.shape)
        states, probabilities = IMMFilter().filter(measurements)
        raw_error = np.mean(np.linalg.norm(measurements[5:] - truth[5:], axis=1))
        filtered_error = np.mean(np.linalg.norm(states[5:, :2] - truth[5:], axis=1))
        self.assertLess(filtered_error, raw_error)
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))


if __name__ == "__main__":
    unittest.main()
