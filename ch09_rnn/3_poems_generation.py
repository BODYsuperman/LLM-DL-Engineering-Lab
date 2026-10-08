import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import re


# ============================================================
# 1. 处理诗歌数据
# ============================================================

def process_poems(file_path):
    poems = []
    char_set = set()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 去掉标点符号和首尾空格
            line = re.sub(r"[，。、？！：]", "", line).strip()

            if not line:
                continue

            # 保存每一行的字符
            char_set.update(line)
            poems.append(list(line))

    # 字符表
    vocab = sorted(char_set) + ["<UNK>"]

    # 字符 -> ID
    word2idx = {
        word: idx
        for idx, word in enumerate(vocab)
    }

    # 未知字符 ID
    unk_id = word2idx["<UNK>"]

    # 每一行诗转换成数字序列
    sequences = []

    for poem in poems:
        seq = [
            word2idx.get(word, unk_id)
            for word in poem
        ]

        sequences.append(seq)

    return sequences, word2idx, vocab


# ============================================================
# 2. 读取数据
# ============================================================

id_seq, word2idx, vocab = process_poems("../data/poems.txt")

print("数据处理完成")
print("dataset 中原始诗句数量:", len(id_seq))
print("vocab size:", len(vocab))
print("word2idx size:", len(word2idx))


# ============================================================
# 3. Dataset
# ============================================================

class PoetryDataset(Dataset):

    def __init__(self, sequences, seq_len):

        self.seq_len = seq_len
        self.data = []

        for seq in sequences:

            # 至少需要 seq_len + 1 个字符
            # 才能构造 X -> Y
            if len(seq) <= self.seq_len:
                continue

            # 滑动窗口
            for i in range(len(seq) - self.seq_len):

                # 输入
                x = seq[i:i + self.seq_len]

                # 标签
                # 比 x 向后移动一个字符
                y = seq[i + 1:i + 1 + self.seq_len]

                self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        x = torch.tensor(
            self.data[idx][0],
            dtype=torch.long
        )

        y = torch.tensor(
            self.data[idx][1],
            dtype=torch.long
        )

        return x, y


# ============================================================
# 4. 创建 Dataset
# ============================================================

seq_len = 30

dataset = PoetryDataset(
    id_seq,
    seq_len
)

print("训练样本数量:", len(dataset))


# ============================================================
# 5. RNN 模型
# ============================================================

class PoetryRNN(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_size=256,
        num_layers=1
    ):

        super(PoetryRNN, self).__init__()

        # 字符 ID -> embedding
        self.embed = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        # RNN
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # hidden state -> vocab_size
        self.linear = nn.Linear(
            in_features=hidden_size,
            out_features=vocab_size
        )

    def forward(self, input, hx=None):

        # [N, L]
        embed = self.embed(input)

        # [N, L, embedding_dim]
        output, hidden = self.rnn(
            embed,
            hx
        )

        # [N, L, vocab_size]
        output = self.linear(output)

        return output, hidden


# ============================================================
# 6. 设置设备
# ============================================================

if torch.cuda.is_available():

    device = torch.device("cuda")

elif torch.backends.mps.is_available():

    device = torch.device("mps")

else:

    device = torch.device("cpu")


print("Using device:", device)


# ============================================================
# 7. 创建模型
# ============================================================

embedding_dim = 256
hidden_size = 512
num_layers = 2

model = PoetryRNN(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    hidden_size=hidden_size,
    num_layers=num_layers
).to(device)


# ============================================================
# 8. 训练
# ============================================================

def train(
    model,
    dataset,
    lr,
    epoch_num,
    batch_size,
    device
):

    model.train()

    # Dataset -> DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Cross Entropy
    loss_fn = nn.CrossEntropyLoss()

    # Adam
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )

    for epoch in range(epoch_num):

        loss_accumulate = 0

        for batch_idx, (x, y) in enumerate(dataloader):

            # -----------------------------
            # 1. 数据放到设备
            # -----------------------------

            x = x.to(device)
            y = y.to(device)

            # x:
            # [batch_size, seq_len]

            # y:
            # [batch_size, seq_len]


            # -----------------------------
            # 2. Forward
            # -----------------------------

            output, _ = model(x)

            # output:
            # [batch_size, seq_len, vocab_size]


            # -----------------------------
            # 3. 计算 Loss
            # -----------------------------

            # CrossEntropyLoss 要求：
            #
            # [N, C, L]
            #
            # 所以：
            #
            # [N, L, C]
            #      ↓
            # [N, C, L]

            loss_value = loss_fn(
                output.transpose(1, 2),
                y
            )


            # -----------------------------
            # 4. 清空旧梯度
            # -----------------------------

            optimizer.zero_grad()


            # -----------------------------
            # 5. 反向传播
            # -----------------------------

            loss_value.backward()


            # -----------------------------
            # 6. 更新参数
            # -----------------------------

            optimizer.step()


            # -----------------------------
            # 7. 累计 Loss
            # -----------------------------

            loss_accumulate += loss_value.item()


            # -----------------------------
            # 8. 显示进度
            # -----------------------------

            progress = int(
                (batch_idx + 1)
                / len(dataloader)
                * 50
            )

            print(
                f"\rEpoch:{epoch + 1:0>3} "
                f"[{'=' * progress}]",
                end=""
            )


        # 一个 epoch 的平均 Loss
        this_loss = (
            loss_accumulate
            / len(dataloader)
        )

        print(
            f" this_loss: {this_loss:.6f}"
        )


    return optimizer


# ============================================================
# 9. 超参数
# ============================================================

lr = 1e-3
epoch_num = 200
batch_size = 64


# ============================================================
# 10. 开始训练
# ============================================================

optimizer = train(
    model,
    dataset,
    lr,
    epoch_num,
    batch_size,
    device
)


# ============================================================
# 11. 保存模型
# ============================================================

checkpoint = {

    # 模型参数
    "model_state_dict": model.state_dict(),

    # optimizer 参数
    "optimizer_state_dict": optimizer.state_dict(),

    # 词表
    "vocab": vocab,

    # 字符 -> ID
    "word2idx": word2idx,

    # 模型结构参数
    "embedding_dim": embedding_dim,
    "hidden_size": hidden_size,
    "num_layers": num_layers,

    # 数据参数
    "seq_len": seq_len,

    # 训练参数
    "lr": lr,
    "epoch_num": epoch_num,
    "batch_size": batch_size
}


torch.save(
    checkpoint,
    "poetry_rnn.pth"
)

print()
print("模型保存完成: poetry_rnn.pth")

# =========================
# 1. 加载 checkpoint
# =========================
checkpoint = torch.load(
    "models/poetry_rnn.pth",
    map_location=device
)

vocab = checkpoint["vocab"]
word2idx = checkpoint["word2idx"]

# =========================
# 2. 创建模型
# =========================
model = PoetryRNN(
    vocab_size=len(vocab),
    embedding_dim=checkpoint["embedding_dim"],
    hidden_size=checkpoint["hidden_size"],
    num_layers=checkpoint["num_layers"]
).to(device)

# =========================
# 3. 加载训练好的参数
# =========================
model.load_state_dict(checkpoint["model_state_dict"])

# 推理模式
model.eval()


# =========================
# 4. 测试生成
# =========================
def generate_poem(model, word2idx, vocab, start_token,
                  line_num=2, line_length=7):

    model.eval()

    poem = []

    # 起始字 → ID
    start_id = word2idx.get(
        start_token,
        word2idx["<UNK>"]
    )

    # 如果起始字存在
    if start_token in word2idx:
        poem.append(start_token)
        current_line_length = line_length - 1
    else:
        current_line_length = line_length

    # [1, 1]
    input = torch.tensor(
        [[start_id]],
        dtype=torch.long,
        device=device
    )

    hidden = None

    with torch.no_grad():

        for _ in range(line_num):

            # 生成逗号前的字符
            while current_line_length > 0:

                output, hidden = model(
                    input,
                    hidden
                )

                # [1, 1, vocab_size]
                # ↓
                # [vocab_size]
                logits = output[0, 0]

                # logits → probability
                prob = torch.softmax(
                    logits,
                    dim=-1
                )

                # 根据概率随机选择一个字
                next_token = torch.multinomial(
                    prob,
                    num_samples=1
                )

                next_id = next_token.item()

                poem.append(vocab[next_id])

                # 下一次输入刚刚生成的字
                input = next_token.unsqueeze(0)

                current_line_length -= 1

            poem.append("，")

            # 第二个半句
            current_line_length = line_length

            while current_line_length > 0:

                output, hidden = model(
                    input,
                    hidden
                )

                logits = output[0, 0]

                prob = torch.softmax(
                    logits,
                    dim=-1
                )

                next_token = torch.multinomial(
                    prob,
                    num_samples=1
                )

                next_id = next_token.item()

                poem.append(vocab[next_id])

                input = next_token.unsqueeze(0)

                current_line_length -= 1

            poem.append("。\n")

            current_line_length = line_length

    return "".join(poem)


# =========================
# 5. 开始测试
# =========================

if __name__ == "__main__":

    test_words = ["床", "春", "山", "月", "风"]
    for start in test_words:

        print("起始字:", start)
        poem = generate_poem(model=model, word2idx=word2idx, vocab=vocab, start_token=start, line_num=2, line_length=7)
        print(poem)