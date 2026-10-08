# LLM-DL-Engineering-Lab

**English** | [简体中文](README.zh-CN.md)

> A hands-on deep learning lab: starting from neural networks written from scratch in pure NumPy, progressing to PyTorch, implementing forward propagation, backpropagation, optimizers, CNNs, and RNNs chapter by chapter.

This is a personal learning/experimentation repository. Each directory corresponds to one chapter of study material, containing numbered standalone scripts and notebooks. There are no build artifacts, test suites, or deployment pipelines — every example runs directly and prints or plots its results.


## 🎯 Learning Goals

- Implement forward propagation, numerical gradients, and backpropagation from scratch in NumPy without any framework, and understand the underlying math
- Master core PyTorch components: Tensor, autograd, `nn.Module`/`Sequential`, `Dataset`/`DataLoader`
- Practice complete training pipelines of typical architectures (MLP, CNN, RNN) on classification, regression, and sequence generation tasks

## 📚 Learning Roadmap

### NumPy from Scratch (ch02–ch05)

- [x] Simple neural network and digit recognizer (batch inference, joblib model persistence)
- [x] Derivatives, numerical gradients, and gradient descent
- [x] Two-layer network training (numerical-gradient version vs. backpropagation version, side by side)
- [x] Layer-based implementation (ReLU/Sigmoid/Affine/Softmax-with-Loss)
- [x] Optimizer comparison (SGD / Momentum / AdaGrad / Adam)

### PyTorch Fundamentals (ch06–ch07)

- [x] Tensor creation, conversion, computation, statistics, indexing
- [x] Automatic differentiation (`grad` vs. `detach` vs. `data`)
- [x] Linear regression (TensorDataset + DataLoader + nn + optim)
- [x] Dropout, `nn.Sequential`
- [x] Activation functions (Sigmoid / Tanh / ReLU / Softmax)
- [x] Loss functions (BCE / CrossEntropy / regression losses)
- [x] Optimizers and LR scheduling (Momentum / StepLR / MultiStepLR / ExponentialLR / Adagrad / RMSProp / Adam)
- [x] House price prediction (sklearn Pipeline feature engineering + PyTorch regression)

### Computer Vision (ch08)

- [x] Convolution and pooling experiments
- [x] Deep CNN
- [x] Fashion-MNIST image classification

### Sequence Models (ch09)

- [x] Embeddings
- [x] RNN fundamentals
- [x] Character-level RNN for Chinese poetry generation (training + inference)

### Next Steps

- [ ] Transformer / Attention
- [ ] LLM-related topics (the "LLM" part of the repo name has not been started yet)

## 🧪 Experiments

| Chapter | Experiment | Data / Input | Description |
| --- | --- | --- | --- |
| ch02 | Digit recognizer | `data/train.csv` (784-dim handwritten digits) | Hand-written forward inference, batch version; model serialized via joblib to `data/nn_sample` |
| ch03/ch04 | Two-layer network training | `data/train.csv` | The same network trained with numerical gradients (ch03) and backpropagation (ch04) for comparison |
| ch05 | Optimizer visualization | Synthetic function `f(x,y)=x²/20+y²` | Plots parameter-update trajectories of four optimizers |
| ch07 | House price prediction | `data/house_prices.csv` | ColumnTransformer (numeric: impute+standardize; categorical: impute+OneHot) + PyTorch regression |
| ch08 | Fashion-MNIST classification | `data/fashion-mnist_{train,test}.csv` | LeNet-style Conv+Sigmoid+Pool architecture |
| ch09 | Poetry generation | `data/poems.txt` (Chinese poetry) | Character-level RNN (Embedding+RNN); weights saved in `poetry_rnn.pth`, with a separate inference script |

## 🛠️ Tech Stack

- Python ≥ 3.14, dependencies and virtualenv managed by [uv](https://docs.astral.sh/uv/)
- NumPy / Matplotlib / scikit-learn / pandas — hand-written NumPy networks and feature engineering
- PyTorch (with torchvision, torchaudio) — all framework-based experiments from ch06 on
- Jupyter Notebook — concept demonstrations
- Also declared: datasets, transformers, tokenizers, accelerate, jieba (dependencies declared but not yet used in depth in ch02–ch09)

## 📂 Project Structure

```text
LLM-DL-Engineering-Lab/
├── ch02_nn_base/        # NumPy simple network, digit recognizer (batch)
├── ch03_train/          # Numerical gradients, gradient descent, two-layer net training
├── ch04_backward/       # Backpropagation version of the two-layer net
├── ch05_optimizer/      # Optimizer trajectory comparison
├── ch06_pytorch_base/   # Tensors, autograd, linear regression
├── ch07_pytorch_dl/     # nn module, Dropout, house price prediction
│   ├── activation_functions/  # Activation function demos
│   ├── loss_functions/        # Loss function demos
│   └── optimizer/             # Optimizer and LR scheduler demos
├── ch08_cnn/            # Convolution/pooling, deep CNN, Fashion-MNIST
├── ch09_rnn/            # Embedding, RNN, poetry generation (train + inference)
├── common/              # Shared NumPy library: functions / gradient / layers / optimizer / load_data
├── data/                # Shared datasets and model files
└── pyproject.toml       # uv project definition
```

`common/` is the support library for ch02–ch05; its code organization follows *Deep Learning from Scratch* (the "fish book"). `two_layer_net.py` exists twice on purpose — ch03 (numerical gradient) and ch04 (backpropagation) — as a pedagogical contrast, not duplicated code.

## 🚀 Getting Started

### Requirements

- Python ≥ 3.14
- Dependencies are declared in `pyproject.toml` and locked in `uv.lock`:

| Category | Packages |
| --- | --- |
| Core DL framework | torch, torchvision, torchaudio |
| Scientific stack | numpy, pandas, matplotlib, seaborn, scikit-learn |
| Jupyter | jupyter, notebook, ipykernel, ipywidgets |
| Model summary | torch-summary, torchsummary |
| NLP / LLM (declared, not yet used in depth) | transformers, tokenizers, datasets, accelerate, jieba |

### Installation

**Option A — uv (recommended, uses the lockfile)**

```bash
uv sync
```

**Option B — Conda**

```bash
conda create -n llm-dl python=3.14 -y
conda activate llm-dl
# install all dependencies from pyproject.toml (requires pip >= 25.1)
pip install -r pyproject.toml
```

If your pip is older, list the packages explicitly:

```bash
pip install torch torchvision torchaudio numpy pandas matplotlib seaborn \
  scikit-learn jupyter notebook ipykernel ipywidgets torch-summary torchsummary \
  transformers tokenizers datasets accelerate jieba
```

### Running Examples

⚠️ Data paths inside scripts are relative to the **chapter directory** (e.g. `../data/train.csv`), so run scripts from within their chapter directory:

```bash
cd ch04_backward
uv run python 1_digit_recognizer_nn_train_bp.py
```

Scripts in ch03/ch04 also `import common`, which lives at the repo root. PyCharm adds content roots to `sys.path` automatically; from a terminal you need:

```bash
cd ch04_backward
PYTHONPATH=.. uv run python 1_digit_recognizer_nn_train_bp.py
```

Notebooks can be opened and run directly with the repo's `.venv` kernel.

## 📝 Notes

- File numbering indicates the suggested reading/running order
- Code comments mix Chinese and English — study-notes style
- Scripts are independent of each other, print or plot results directly, and take no CLI arguments
- This directory is currently not a git repository

## 📖 References

- *Deep Learning from Scratch* (《深度学习入门：基于 Python 的理论与实现》) — the `common/` library structure and two-layer-net implementation pattern in ch02–ch05
- Other books/courses: ...
