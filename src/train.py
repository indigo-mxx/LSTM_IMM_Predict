"""训练 IMM+LSTM 邻居节点位置预测模型。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import generate_dataset
from .model import PositionPredictor


def standardize_fit(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    """使用训练集统计量标准化输入与标签，并保留所需参数。"""
    feature_mean, feature_std = features.mean(axis=(0, 1)), features.std(axis=(0, 1)) + 1e-6
    label_mean, label_std = labels.mean(axis=0), labels.std(axis=0) + 1e-6
    metadata = {
        "feature_mean": feature_mean.tolist(), "feature_std": feature_std.tolist(),
        "label_mean": label_mean.tolist(), "label_std": label_std.tolist(),
    }
    return (features - feature_mean) / feature_std, (labels - label_mean) / label_std, metadata


def standardize_apply(features: np.ndarray, labels: np.ndarray, metadata: dict[str, list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """按训练集统计量处理验证集。"""
    return (
        (features - np.asarray(metadata["feature_mean"])) / np.asarray(metadata["feature_std"]),
        (labels - np.asarray(metadata["label_mean"])) / np.asarray(metadata["label_std"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 IMM+LSTM 邻居节点位置预测模型")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--trajectories", type=int, default=120)
    parser.add_argument("--trajectory-length", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features, labels = generate_dataset(args.trajectories, args.trajectory_length, args.sequence_length, args.seed)
    split = int(len(features) * 0.8)
    train_x, train_y, metadata = standardize_fit(features[:split], labels[:split])
    validation_x, validation_y = standardize_apply(features[split:], labels[split:], metadata)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(TensorDataset(torch.from_numpy(validation_x), torch.from_numpy(validation_y)), batch_size=args.batch_size)

    model = PositionPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    best_validation = float("inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        total_loss, count = 0.0, 0
        with torch.no_grad():
            for batch_x, batch_y in validation_loader:
                value = criterion(model(batch_x.to(device)), batch_y.to(device)).item()
                total_loss += value * len(batch_x)
                count += len(batch_x)
        validation_loss = total_loss / count
        print(f"第 {epoch:02d}/{args.epochs} 轮，验证 MSE：{validation_loss:.5f}")
        if validation_loss < best_validation:
            best_validation = validation_loss
            torch.save(model.state_dict(), args.output_dir / "best_model.pth")

    (args.output_dir / "normalization.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"训练完成，最佳验证 MSE：{best_validation:.5f}；模型已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
