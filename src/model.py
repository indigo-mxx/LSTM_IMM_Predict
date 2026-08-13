"""融合 IMM 特征的 LSTM 位置预测网络。"""

from __future__ import annotations

import torch
from torch import nn


class PositionPredictor(nn.Module):
    """读取时间窗内的观测与 IMM 状态，输出下一时刻二维位置。"""

    def __init__(self, input_size: int = 6, hidden_size: int = 64, layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """预测每个序列对应的下一时刻坐标。"""
        encoded, _ = self.lstm(sequence)
        return self.head(encoded[:, -1, :])
