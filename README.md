# IMM + LSTM 邻居节点位置预测

本项目实现一个可复现实验：先使用交互多模型（IMM）融合匀速与机动两种运动假设，再将观测值与 IMM 状态送入 LSTM，预测邻居节点下一时刻的二维位置。

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --epochs 30
python -m src.predict
```

训练产物默认保存在 `artifacts/`：包括模型权重、归一化参数和预测示意图。

## 项目结构

- `src/imm.py`：二维运动的 IMM 与卡尔曼滤波实现。
- `src/data.py`：带匀速、转弯和加减速阶段的邻居轨迹仿真。
- `src/model.py`：融合 IMM 输出的 LSTM 预测网络。
- `src/train.py`：训练入口。
- `src/predict.py`：加载模型并执行预测与绘图。
- `tests/`：IMM 和端到端数据流测试。

## 方法说明

每个时刻的输入特征为 `[观测位置, IMM 融合位置, IMM 融合速度]`，长度为 `sequence_length` 的时间窗输入 LSTM。标签是窗口后下一时刻的真实二维位置。IMM 通过模型概率自动平衡：

- 平稳模型：较小的过程噪声，适用于匀速移动；
- 机动模型：较大的过程噪声，适用于转弯、加速等轨迹变化。

数据集是可控的合成邻居节点轨迹，便于在没有现场定位数据时完成验证。接入真实数据时，只需将 `(时间, x, y)` 序列转换为 `Trajectory`，复用相同的特征构造和训练流程。

## 验证

```bash
python -m unittest discover -s tests -v
```

## 注意

此项目用于算法原型与简历展示。实际网络节点定位场景应按采样间隔、定位误差分布和运动先验重新标定 IMM 噪声参数，并用真实轨迹重新训练 LSTM。
