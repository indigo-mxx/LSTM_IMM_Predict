"""使用训练好的 IMM+LSTM 模型完成单条邻居轨迹的下一位置预测。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager

from .data import imm_features, simulate_trajectory
from .model import PositionPredictor


def configure_chinese_font() -> None:
    candidate_fonts = [
        Path("/mnt/c/Windows/Fonts/msyh.ttc"),
        Path("/mnt/c/Windows/Fonts/simhei.ttf"),
    ]
    for font_path in candidate_fonts:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    parser = argparse.ArgumentParser(description="演示 IMM+LSTM 的邻居节点位置预测")
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    model_path, normalization_path = args.model_dir / "best_model.pth", args.model_dir / "normalization.json"
    if not model_path.exists() or not normalization_path.exists():
        raise FileNotFoundError("未找到训练产物，请先运行 python -m src.train")

    metadata = json.loads(normalization_path.read_text(encoding="utf-8"))
    trajectory = simulate_trajectory(args.sequence_length + 1, np.random.default_rng(args.seed))
    features = imm_features(trajectory.measurements)
    normalized = (features[-args.sequence_length:] - np.asarray(metadata["feature_mean"])) / np.asarray(metadata["feature_std"])
    model = PositionPredictor()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)).squeeze(0).numpy()
    prediction = prediction * np.asarray(metadata["label_std"]) + np.asarray(metadata["label_mean"])
    target = trajectory.truth[-1]
    error = np.linalg.norm(prediction - target)
    print(f"真实下一位置：[{target[0]:.2f}, {target[1]:.2f}]")
    print(f"预测下一位置：[{prediction[0]:.2f}, {prediction[1]:.2f}]")
    print(f"欧氏误差：{error:.3f}")

    configure_chinese_font()
    plt.figure(figsize=(7, 5))
    plt.plot(trajectory.truth[:, 0], trajectory.truth[:, 1], "-o", label="真实轨迹")
    plt.scatter(trajectory.measurements[:, 0], trajectory.measurements[:, 1], marker="x", label="定位观测")
    plt.scatter(*prediction, s=100, marker="*", label="LSTM 预测")
    plt.axis("equal")
    plt.legend()
    plt.title("IMM+LSTM 邻居节点下一位置预测")
    plt.tight_layout()
    plt.savefig(args.model_dir / "prediction.png", dpi=160)
    print(f"示意图已保存到 {args.model_dir / 'prediction.png'}")


if __name__ == "__main__":
    main()
