import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import time

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "get_batch",
    "create_mini_dataset",
    "NanoTransformer",
]


# 超参数 - 极度简化
vocab_size = 100  # 极小的词汇表
n_embd = 32  # 微小的嵌入维度
n_head = 2  # 少量注意力头
n_layer = 2  # 少量层
context_length = 16  # 很短的上下文
batch_size = 8  # 小批量
max_iters = 50  # 极少的迭代次数
learning_rate = 0.01
device = "cpu"  # 明确使用CPU


def get_batch(
    x: np.ndarray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    生成训练批次数据

    Args:
        x: 令牌ID的一维numpy数组
        batch_size: 批次大小
        context_length: 序列长度
        device: 目标设备 (如 'cpu' 或 'cuda:0')

    Returns:
        (inputs,targets): (输入序列, 下一个目标序列) (batch_size, context_length)
    """
    # 确保数据足够长 (至少需要 context_length+1 个令牌)
    assert len(x) > context_length + 1, "数据长度不足"

    # 随机生成起始位置 (避免越界)
    max_start = len(x) - context_length - 1
    starts = np.random.randint(0, max_start, size=batch_size)

    # 提取输入序列和目标序列,生成批次 (batch_size, context_length)
    # targets 是 inputs 的下一个令牌，用于交叉熵损失预测
    inputs = np.stack([x[i : i + context_length] for i in starts])
    targets = np.stack([x[i + 1 : i + context_length + 1] for i in starts])

    # 转为PyTorch张量并移动到指定设备
    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


def save_checkpoint(
    model: nn.Module, optimizer: optim.Optimizer, iteration: int, out: str
):
    """
    保存模型检查点

    Args:
        model: 要保存的模型
        optimizer: 优化器
        iteration: 当前训练迭代次数
        out: 保存目标 (文件路径/文件对象)
    """
    # 保存模型状态、优化器状态和迭代次数
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(src: str, model: nn.Module, optimizer: optim.Optimizer) -> int:
    """
    加载模型检查点

    Args:
        src: 检查点源 (文件路径/文件对象)
        model: 待恢复的模型
        optimizer: 待恢复的优化器

    Returns:
        保存的迭代次数
    """
    # 加载检查点数据
    checkpoint = torch.load(src)

    # 恢复模型和优化器状态
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["iteration"]


# 生成微型数据集 (1000个token)
def create_mini_dataset():
    return np.random.randint(0, vocab_size, size=1000)


# 简化版Transformer模型
class NanoTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(
            vocab_size,
            n_embd,
        )
        self.position_embedding = nn.Embedding(context_length, n_embd)
        self.blocks = nn.Sequential(
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=4 * n_embd,
                batch_first=True,
            ),
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=4 * n_embd,
                batch_first=True,
            ),
        )
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: 输入令牌ID张量 (batch_size, context_length)

        Returns:
            logits: 输出预测张量 (batch_size, context_length, vocab_size)
        """
        B, T = idx.shape
        token_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        logits = self.head(x)
        return logits


if __name__ == "__main__":
    # 创建微型数据集
    train_data = create_mini_dataset()
    val_data = create_mini_dataset()

    print(f"训练数据: {len(train_data)} tokens, 验证数据: {len(val_data)} tokens")

    # 初始化模型并移动到CPU
    model = NanoTransformer().to(device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    start_time = time.time()
    print("开始训练...")

    for iter_num in range(max_iters):
        # 使用提供的get_batch函数获取数据
        inputs, targets = get_batch(train_data, batch_size, context_length, device)

        # 前向传播
        logits: torch.Tensor = model(inputs)
        loss: torch.Tensor = criterion(logits.view(-1, vocab_size), targets.view(-1))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每10次迭代打印一次进度
        if iter_num % 10 == 0:
            elapsed = time.time() - start_time
            print(
                f"迭代 {iter_num}/{max_iters} | 损失: {loss.item():.4f} | 用时: {elapsed:.1f}s"
            )

    # 最终评估
    model.eval()
    with torch.no_grad():
        inputs, targets = get_batch(val_data, batch_size, context_length, device)
        logits = model(inputs)
        val_loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        print(f"\n训练完成! 最终验证损失: {val_loss.item():.4f}")
        print(f"总训练时间: {time.time() - start_time:.1f}秒")

        # 演示保存检查点（可选）
        save_checkpoint(model, optimizer, max_iters, "./out/mini_model_checkpoint.pt")
        print("已保存微型模型检查点: ./out/mini_model_checkpoint.pt")
