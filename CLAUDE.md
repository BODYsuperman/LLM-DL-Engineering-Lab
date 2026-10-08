# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

Personal deep-learning study lab, organized as one directory per book chapter (`ch02_nn_base` … `ch09_rnn`). Chapters 2–5 implement neural networks from scratch in NumPy; chapters 6–9 switch to PyTorch (tensors, nn module, CNN, RNN). Each chapter contains numbered, self-contained scripts/notebooks that are run individually — there is no package to build, no test suite, and no lint/CI config (Black is configured only via PyCharm).

## Environment

- Python ≥ 3.14 managed by `uv` (see `pyproject.toml` / `uv.lock`, interpreter in `.venv`).
- Install/sync deps: `uv sync`. Run something: `uv run python <script>` (PyCharm run configs also work).

## Running Scripts — Path Conventions (important)

- Scripts load data via paths **relative to the chapter directory**, e.g. `pd.read_csv("../data/train.csv")`, `joblib.load("../data/nn_sample")`. Run them with the chapter directory as the working directory (`cd ch04_backward && uv run python 1_digit_recognizer_nn_train_bp.py`), or paths will break.
- Ch03/ch04 scripts do `from common.load_data import get_data` and `from two_layer_net import TwoLayerNet`. `common/` lives at the repo root, so the **repo root must also be on `sys.path`** (PyCharm adds content roots automatically; from a terminal, `PYTHONPATH=..` works). No script adds this itself.
- Not a git repository — no commits/branches to worry about.

## Architecture

- `common/` — NumPy support library for ch02–ch05, modeled on *Deep Learning from Scratch*: `functions.py` (activations + derivatives, cross-entropy), `gradient.py` (numerical gradient), `layers.py` (Relu/Sigmoid/Affine/Softmax-with-loss layer classes with forward/backward), `optimizer.py` (SGD, Momentum, etc.), `load_data.py` (`get_data()` loads `data/train.csv`, MinMax-scales, 70/30 split, returns `train_x, train_y, test_x, test_y`).
- `two_layer_net.py` exists **twice, deliberately different**: `ch03_train/` computes gradients numerically (`numerical_gradient`); `ch04_backward/` implements backpropagation (`gradient()` via layer backward passes). Don't "unify" them — the duplication is the pedagogical point. Later chapters repeat the same network with optimizers/backprop variations.
- `data/` — shared datasets: `train.csv` (784-feature digit classification), `fashion-mnist_{train,test}.csv`, `house_prices.csv` (+ `house_prices_cols.txt`), `poems.txt` (Chinese poetry, UTF-8, used by ch09 RNN), `nn_sample` (joblib-serialized NumPy network from ch02), `duck.jpg`.
- ch07 has topical subdirectories (`activation_functions/`, `loss_functions/`, `optimizer/`), each a set of numbered demo scripts.
- ch09 keeps trained weights in `poetry_rnn.pth`.

## Conventions

- Files are numbered by intended reading/execution order within each chapter.
- Comments mix Chinese and English; keep that style when editing existing files.
- Each script is standalone and prints/plots results directly (matplotlib); none define `main()` or CLI args.
