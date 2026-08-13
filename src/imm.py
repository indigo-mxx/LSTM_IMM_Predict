"""二维位置观测下的交互多模型（IMM）滤波器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


Array = np.ndarray


def _gaussian_likelihood(innovation: Array, covariance: Array) -> float:
    """计算零均值高斯分布下创新量的似然值。"""
    dimension = innovation.shape[0]
    sign, log_det = np.linalg.slogdet(covariance)
    if sign <= 0:
        return 1e-12
    mahalanobis = innovation @ np.linalg.solve(covariance, innovation)
    log_prob = -0.5 * (dimension * np.log(2.0 * np.pi) + log_det + mahalanobis)
    return float(np.exp(np.clip(log_prob, -700, 50)))


@dataclass
class KalmanModel:
    """恒速度状态模型；通过不同过程噪声表示不同运动模式。"""

    process_noise: float
    measurement_noise: float
    dt: float = 1.0

    def matrices(self) -> Tuple[Array, Array, Array, Array]:
        """返回状态转移、过程噪声、观测和观测噪声矩阵。"""
        dt = self.dt
        transition = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        base_noise = np.array(
            [[dt**4 / 4, 0.0, dt**3 / 2, 0.0], [0.0, dt**4 / 4, 0.0, dt**3 / 2],
             [dt**3 / 2, 0.0, dt**2, 0.0], [0.0, dt**3 / 2, 0.0, dt**2]],
            dtype=np.float64,
        )
        process_covariance = self.process_noise * base_noise
        observation = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        measurement_covariance = np.eye(2, dtype=np.float64) * self.measurement_noise
        return transition, process_covariance, observation, measurement_covariance


class IMMFilter:
    """由平稳与机动两个卡尔曼模型组成的 IMM 滤波器。"""

    def __init__(
        self,
        models: Iterable[KalmanModel] | None = None,
        transition_probabilities: Array | None = None,
        initial_probabilities: Array | None = None,
    ) -> None:
        self.models = list(models or [KalmanModel(0.08, 1.0), KalmanModel(1.5, 1.0)])
        self.model_count = len(self.models)
        self.transition_probabilities = np.asarray(
            transition_probabilities if transition_probabilities is not None else [[0.94, 0.06], [0.10, 0.90]],
            dtype=np.float64,
        )
        self.probabilities = np.asarray(
            initial_probabilities if initial_probabilities is not None else np.ones(self.model_count) / self.model_count,
            dtype=np.float64,
        )
        self.states = np.zeros((self.model_count, 4), dtype=np.float64)
        self.covariances = np.repeat(np.eye(4, dtype=np.float64)[None, :, :] * 10.0, self.model_count, axis=0)
        self._initialized = False

    def reset(self, position: Array | None = None) -> None:
        """重置滤波器；若给定位置，则以其作为初始位置。"""
        self.states.fill(0.0)
        if position is not None:
            self.states[:, :2] = np.asarray(position, dtype=np.float64)
        self.covariances = np.repeat(np.eye(4, dtype=np.float64)[None, :, :] * 10.0, self.model_count, axis=0)
        self.probabilities[:] = 1.0 / self.model_count
        self._initialized = position is not None

    def _mix_states(self) -> tuple[Array, Array, Array]:
        """按模式转移概率混合上一步状态与协方差。"""
        predicted_mode_probabilities = self.transition_probabilities.T @ self.probabilities
        mixing = (self.transition_probabilities * self.probabilities[:, None]) / predicted_mode_probabilities[None, :]
        mixed_states = mixing.T @ self.states
        mixed_covariances = np.zeros_like(self.covariances)
        for destination in range(self.model_count):
            for source in range(self.model_count):
                delta = self.states[source] - mixed_states[destination]
                mixed_covariances[destination] += mixing[source, destination] * (
                    self.covariances[source] + np.outer(delta, delta)
                )
        return mixed_states, mixed_covariances, predicted_mode_probabilities

    def step(self, measurement: Array) -> tuple[Array, Array]:
        """融合一条二维观测，返回融合状态和当前各模式概率。"""
        measurement = np.asarray(measurement, dtype=np.float64)
        if measurement.shape != (2,):
            raise ValueError("观测值必须为形如 [x, y] 的二维向量")
        if not self._initialized:
            self.reset(measurement)
            return self.fused_state(), self.probabilities.copy()

        mixed_states, mixed_covariances, predicted_probabilities = self._mix_states()
        likelihoods = np.empty(self.model_count, dtype=np.float64)
        new_states = np.empty_like(self.states)
        new_covariances = np.empty_like(self.covariances)
        for index, model in enumerate(self.models):
            transition, process_covariance, observation, measurement_covariance = model.matrices()
            state_prediction = transition @ mixed_states[index]
            covariance_prediction = transition @ mixed_covariances[index] @ transition.T + process_covariance
            innovation = measurement - observation @ state_prediction
            innovation_covariance = observation @ covariance_prediction @ observation.T + measurement_covariance
            gain = np.linalg.solve(innovation_covariance, observation @ covariance_prediction).T
            new_states[index] = state_prediction + gain @ innovation
            identity = np.eye(4)
            updated_covariance = (identity - gain @ observation) @ covariance_prediction
            new_covariances[index] = updated_covariance @ (identity - gain @ observation).T + gain @ measurement_covariance @ gain.T
            likelihoods[index] = _gaussian_likelihood(innovation, innovation_covariance)

        posterior = predicted_probabilities * likelihoods
        normalizer = posterior.sum()
        self.probabilities = posterior / normalizer if normalizer > 1e-15 else np.ones(self.model_count) / self.model_count
        self.states, self.covariances = new_states, new_covariances
        return self.fused_state(), self.probabilities.copy()

    def fused_state(self) -> Array:
        """返回以模式后验概率加权的状态估计。"""
        return self.probabilities @ self.states

    def filter(self, measurements: Array) -> tuple[Array, Array]:
        """过滤一组观测，返回每时刻融合状态和模式概率。"""
        measurements = np.asarray(measurements, dtype=np.float64)
        self.reset()
        states, probabilities = [], []
        for measurement in measurements:
            state, probability = self.step(measurement)
            states.append(state)
            probabilities.append(probability)
        return np.asarray(states), np.asarray(probabilities)
