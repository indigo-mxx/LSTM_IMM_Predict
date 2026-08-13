"""合成邻居节点轨迹，并构建带 IMM 特征的序列数据集。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .imm import IMMFilter


@dataclass
class Trajectory:
    """一条邻居节点的真实位置与带噪观测。"""

    truth: np.ndarray
    measurements: np.ndarray


def simulate_trajectory(length: int, rng: np.random.Generator, measurement_std: float = 0.8) -> Trajectory:
    """生成包含匀速、机动和加减速阶段的二维移动轨迹。"""
    if length < 8:
        raise ValueError("轨迹长度至少为 8")
    position = rng.uniform(-20.0, 20.0, size=2)
    angle = rng.uniform(-np.pi, np.pi)
    speed = rng.uniform(0.8, 2.5)
    velocity = speed * np.array([np.cos(angle), np.sin(angle)])
    truth = np.zeros((length, 2), dtype=np.float64)
    acceleration = np.zeros(2, dtype=np.float64)
    for time_index in range(length):
        if time_index % rng.integers(7, 14) == 0:
            turn = rng.normal(0.0, 0.22)
            rotation = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
            velocity = rotation @ velocity
            acceleration = rng.normal(0.0, 0.05, size=2)
        velocity = np.clip(velocity + acceleration, -3.5, 3.5)
        position = position + velocity
        truth[time_index] = position
    measurements = truth + rng.normal(0.0, measurement_std, size=truth.shape)
    return Trajectory(truth=truth, measurements=measurements)


def imm_features(measurements: np.ndarray) -> np.ndarray:
    """将观测转换为 LSTM 输入：观测位置、IMM 位置和 IMM 速度。"""
    filter_ = IMMFilter()
    states, _ = filter_.filter(measurements)
    return np.concatenate([measurements, states], axis=1).astype(np.float32)


def build_dataset(trajectories: Iterable[Trajectory], sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    """滑动窗口构建样本，标签为下一时刻的真实位置。"""
    features, labels = [], []
    for trajectory in trajectories:
        trajectory_features = imm_features(trajectory.measurements)
        for start in range(0, len(trajectory.truth) - sequence_length):
            end = start + sequence_length
            features.append(trajectory_features[start:end])
            labels.append(trajectory.truth[end])
    if not features:
        raise ValueError("轨迹数量或长度不足，无法构建序列样本")
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def generate_dataset(
    trajectory_count: int,
    trajectory_length: int,
    sequence_length: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """生成可复现的样本集。"""
    rng = np.random.default_rng(seed)
    trajectories = [simulate_trajectory(trajectory_length, rng) for _ in range(trajectory_count)]
    return build_dataset(trajectories, sequence_length)
