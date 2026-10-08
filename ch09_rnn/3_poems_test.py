import torch
from torch import nn

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

if torch.cuda.is_available():

    device = torch.device("cuda")

elif torch.backends.mps.is_available():

    device = torch.device("mps")

else:

    device = torch.device("cpu")


print("Using device:", device)


# =========================
# 1. 加载 checkpoint
# =========================
checkpoint = torch.load(
    "poetry_rnn.pth",
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