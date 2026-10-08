# LLM-DL-Engineering-Lab

[English](README.md) | **简体中文**

> 深度学习入门实验仓库：从纯 NumPy 手写神经网络开始，逐步过渡到 PyTorch，按章节递进地实现前向传播、反向传播、优化器、CNN 与 RNN。

这是一个个人学习/实验型仓库，每个目录对应一章学习内容，包含按序号编排的独立脚本与 Notebook。没有构建产物、测试套件或部署流程——所有示例直接运行并打印/绘图输出结果。

## 🎯 学习目标

- 不依赖框架，用 NumPy 从零实现神经网络的前向传播、数值梯度与反向传播，理解其数学原理
- 掌握 PyTorch 核心组件：Tensor、autograd、`nn.Module`/`Sequential`、`Dataset`/`DataLoader`
- 实践典型网络结构（MLP、CNN、RNN）在分类、回归、序列生成任务上的完整训练流程

## 📚 学习路线图

### NumPy 从零实现（ch02–ch05）

- [x] 简单神经网络与数字识别（含批量推理、joblib 模型持久化）
- [x] 导数、数值梯度与梯度下降
- [x] 两层网络训练（数值梯度版 vs 反向传播版对照实现）
- [x] 层化实现（ReLU/Sigmoid/Affine/Softmax-with-Loss）
- [x] 优化器对比（SGD / Momentum / AdaGrad / Adam）

### PyTorch 基础（ch06–ch07）

- [x] Tensor 创建、转换、运算、统计、索引
- [x] 自动求导（`grad`、`detach`、`data` 的区别）
- [x] 线性回归（TensorDataset + DataLoader + nn + optim）
- [x] Dropout、`nn.Sequential`
- [x] 激活函数（Sigmoid / Tanh / ReLU / Softmax）
- [x] 损失函数（BCE / CrossEntropy / 回归损失）
- [x] 优化器与学习率调度（Momentum / StepLR / MultiStepLR / ExponentialLR / Adagrad / RMSProp / Adam）
- [x] 房价预测（sklearn Pipeline 特征工程 + PyTorch 回归）

### 计算机视觉（ch08）

- [x] 卷积、池化原理实验
- [x] 深层 CNN
- [x] Fashion-MNIST 图像分类

### 序列模型（ch09）

- [x] Embedding
- [x] RNN 基本原理
- [x] 字符级 RNN 中文古诗生成（训练 + 推理）

### 后续计划

- [ ] Transformer / Attention
- [ ] LLM 相关主题（与仓库名 "LLM" 对应的部分尚未开始）

## 🧪 实验说明

| 章节 | 实验 | 数据 / 输入 | 说明 |
| --- | --- | --- | --- |
| ch02 | 数字识别器 | `data/train.csv`（784 维手写数字） | 手写前向推理，批量版本；模型经 joblib 序列化到 `data/nn_sample` |
| ch03/ch04 | 两层网络训练 | `data/train.csv` | 同一网络分别用数值梯度（ch03）与反向传播（ch04）训练，对照理解 BP |
| ch05 | 优化器可视化对比 | 合成函数 `f(x,y)=x²/20+y²` | 绘制四种优化器的参数更新轨迹 |
| ch07 | 房价预测 | `data/house_prices.csv` | ColumnTransformer（数值：填充+标准化；类别：填充+OneHot）+ PyTorch 回归 |
| ch08 | Fashion-MNIST 分类 | `data/fashion-mnist_{train,test}.csv` | Conv+Sigmoid+Pool 的类 LeNet 结构 |
| ch09 | 古诗生成 | `data/poems.txt`（中文诗歌） | 字符级 RNN（Embedding+RNN），权重保存于 `poetry_rnn.pth`，含独立推理脚本 |

## 🛠️ 技术栈

- Python ≥ 3.14，[uv](https://docs.astral.sh/uv/) 管理依赖与虚拟环境
- NumPy / Matplotlib / scikit-learn / pandas —— NumPy 手写网络与特征工程
- PyTorch（含 torchvision、torchaudio）—— ch06 起的所有框架实验
- Jupyter Notebook —— 原理演示类内容
- 其他：datasets、transformers、tokenizers、accelerate、jieba（已声明依赖，ch02–ch09 暂未深入使用）

## 📂 项目结构

```text
LLM-DL-Engineering-Lab/
├── ch02_nn_base/        # NumPy 简单网络、数字识别（含批量）
├── ch03_train/          # 数值梯度、梯度下降、两层网络训练
├── ch04_backward/       # 反向传播版两层网络
├── ch05_optimizer/      # 优化器轨迹对比
├── ch06_pytorch_base/   # Tensor、autograd、线性回归
├── ch07_pytorch_dl/     # nn 模块、Dropout、房价预测
│   ├── activation_functions/  # 激活函数演示
│   ├── loss_functions/        # 损失函数演示
│   └── optimizer/             # 优化器与 LR 调度演示
├── ch08_cnn/            # 卷积/池化原理、深层 CNN、Fashion-MNIST
├── ch09_rnn/            # Embedding、RNN、古诗生成（训练+推理）
├── common/              # NumPy 公共库：functions / gradient / layers / optimizer / load_data
├── data/                # 共享数据集与模型文件
└── pyproject.toml       # uv 项目定义
```

其中 `common/` 是 ch02–ch05 的支撑库，代码组织参考《深度学习入门：基于 Python 的理论与实现》（鱼书）。`two_layer_net.py` 在 ch03（数值梯度版）与 ch04（反向传播版）中各有一份，是有意的教学对照，并非重复代码。

## 🚀 快速开始

### 环境要求

- Python ≥ 3.14
- 依赖声明于 `pyproject.toml`，并由 `uv.lock` 锁定版本：

| 类别 | 依赖包 |
| --- | --- |
| 深度学习框架 | torch, torchvision, torchaudio |
| 科学计算 | numpy, pandas, matplotlib, seaborn, scikit-learn |
| Jupyter | jupyter, notebook, ipykernel, ipywidgets |
| 模型结构打印 | torch-summary, torchsummary |
| NLP / LLM（已声明，暂未深入使用） | transformers, tokenizers, datasets, accelerate, jieba |

### 环境安装

**方式 A —— uv（推荐，使用锁文件）**

```bash
uv sync
```

**方式 B —— Conda**

```bash
conda create -n llm-dl python=3.14 -y
conda activate llm-dl
# 从 pyproject.toml 安装全部依赖（需 pip >= 25.1）
pip install -r pyproject.toml
```

若 pip 版本较旧，可显式列出依赖包：

```bash
pip install torch torchvision torchaudio numpy pandas matplotlib seaborn \
  scikit-learn jupyter notebook ipykernel ipywidgets torch-summary torchsummary \
  transformers tokenizers datasets accelerate jieba
```

### 运行示例

⚠️ 脚本内的数据路径相对于**章节目录**（如 `../data/train.csv`），请在章节目录下运行：

```bash
cd ch04_backward
uv run python 1_digit_recognizer_nn_train_bp.py
```

ch03/ch04 的脚本还会 `import common`，而 `common/` 位于仓库根目录。PyCharm 会自动将内容根加入 `sys.path`；在终端运行时需要：

```bash
cd ch04_backward
PYTHONPATH=.. uv run python 1_digit_recognizer_nn_train_bp.py
```

Notebook 直接用仓库 `.venv` 内核打开运行即可。

## 📝 说明

- 文件序号即建议的阅读/运行顺序
- 代码注释为中英混合，属于学习笔记风格
- 各脚本相互独立，直接输出结果或绘图，无 CLI 参数
- 本目录当前不是 git 仓库

## 📖 参考资料

- 《深度学习入门：基于 Python 的理论与实现》（鱼书）—— ch02–ch05 的 `common/` 库结构与两层网络实现范式
- 其他参考书目/课程：...
