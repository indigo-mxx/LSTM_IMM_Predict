"""数据构建测试。"""

import unittest

from src.data import generate_dataset


class DataTest(unittest.TestCase):
    def test_生成序列形状正确(self) -> None:
        features, labels = generate_dataset(trajectory_count=3, trajectory_length=20, sequence_length=6, seed=1)
        self.assertEqual(features.shape, (42, 6, 6))
        self.assertEqual(labels.shape, (42, 2))


if __name__ == "__main__":
    unittest.main()
