"""
此处保存CS336 homework1中 过去用到的函数

"""

from typing import List
from torch import optim
from typing import Optional, Callable, Any

import torch.nn as nn
import numpy as np
import torch
import math


class DataLoader:
    """
    # 作业 5.1 数据加载
    """

    def __init__(
        self, data: List[int], batch_size: int, context_length: int, shuffle=True
    ):
        self.data = data
        self.data_len = len(data)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.context_length = context_length

    def __iter__(self):
        return self.get_valid_batch_data_iter()

    def get_train_batch_data(self):
        """
        随机获取一个batch的数据，y 正好是 x 向左移动一个位置的结果。
        模型在看到 x 的第 i 个位置的输入时，需要努力预测出 y 在第 i 个位置的词元，这正是我们想要的“预测下一个词”的效果。
        """
        if self.shuffle:
            idxs = np.random.randint(
                0, self.data_len - self.context_length - 1, size=(self.batch_size,)
            )  # 随机选择一个batch_size大小的索引,最后的括号其实写不写都行，写上了就更明确输出是一个数组
            x = np.stack([self.data[i : i + self.context_length] for i in idxs])
            y = np.stack([self.data[i + 1 : i + self.context_length + 1] for i in idxs])
            return torch.tensor(x), torch.tensor(y)
        else:
            self.get_valid_batch_data_iter()

    def get_valid_batch_data_iter(self):
        """
        验证集的区别之处在于，不需要随机选择，而是直接从数据集中按顺序获取所有数据，并用迭代器返回
        """
        start_num = (
            self.data_len - self.context_length - 1
        ) // self.batch_size  # 表示有多少个batch
        for i in range(start_num):
            bias = i * self.batch_size  # 表示每一个batch开始的位置
            x = np.stack(
                [
                    self.data[bias : bias + self.context_length]
                    for j in range(self.batch_size)
                ]
            )
            y = np.stack(
                [
                    self.data[bias + 1 : bias + self.context_length + 1]
                    for j in range(self.batch_size)
                ]
            )
            yield torch.tensor(x), torch.tensor(y)

    def __len__(self):
        """
        返回数据集的batch数量
        用法：len(DataLoader) (对象实例)
        """
        return self.data_len // self.batch_size


class EmbeddingModule(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        # 保存词汇表大小和嵌入向量维度
        self.vocab_size = vocab_size  # 词汇表大小
        self.d_model = d_model  # 每个词向量的维度
        self.device = device  # 参数存储的设备（如 CPU/GPU）
        self.dtype = dtype  # 参数的数据类型（如 float32）

        # 初始化嵌入矩阵：形状为 (vocab_size, d_model)
        # 使用 nn.Parameter 包装，使其成为可学习参数
        self.embedding_matrix = nn.Parameter(
            torch.empty(self.vocab_size, self.d_model, device=device, dtype=dtype)
        )

        # 使用截断正态分布初始化参数，均值为0，标准差为1，截断范围为 [-3, 3]
        std = 1
        torch.nn.init.trunc_normal_(
            self.embedding_matrix, std=std, a=-3.0 * std, b=3.0 * std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 使用索引 x 从嵌入矩阵中获取对应的嵌入向量
        return self.embedding_matrix[x]  # x 的形状为 (batch_size, seq_length)


class RMSNorm(nn.Module):
    """
    # 3.4.1 均方根归一化
    Root Mean Square Layer Normalization (RMSNorm) 模块。

    RMSNorm 是归一化技术，它通过将输入除以输入的平方根的平均值来稳定训练。
    公式是：
    x_norm = x / (x.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
    x_norm = x_norm * weight

    Args:
        d_model (int): 经过embedding层之后，每个token的维度
        eps (float): 一个很小的常数，用于避免除以零
        device (torch.device): 设备
        dtype (torch.dtype): 数据类型
    input:
        x: (batch_size, seq_len, d_model) 输入的稠密向量
    output:
        x_norm: (batch_size, seq_len, d_model) 归一化后的稠密向量

    参考论文: https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.eps = eps  # 保存 epsilon 值，用于归一化中的数值稳定

        # 创建可学习的缩放权重参数，形状为 (d_model,)，初始化为全 1
        # 每个特征维度有一个对应的缩放因子
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        重写前向传播函数。

        对输入张量 x 进行 RMSNorm 归一化，然后用可学习的权重进行缩放。

        Args:
            x: 输入张量，形状为 (batch_size, sequence_length, d_model) sequence_length表示输入语句中token(位置)的数量
        Returns:
            torch.Tensor: 输出张量，形状与输入相同，但经过归一化和缩放
        """
        input_dtype = x.dtype
        x = x.to(torch.float32)

        # 计算最后一个维度（特征维度）上的均方值（即方差，但不减均值）
        # shape: (batch_size, sequence_length, 1)
        variance = x.pow(2).mean(-1, keepdim=True)

        # 使用均方根（RMS）进行归一化：x / sqrt(variance + eps)
        # torch.rsqrt 是 1 / sqrt(x) 的高效实现
        x = x * torch.rsqrt(variance + self.eps)

        # 将归一化后的结果与可学习的权重相乘（逐通道缩放）
        # 权重会自动广播到 batch , sequence 维度
        x = self.weight * x

        # 将结果转换回原始输入的数据类型（例如从 float32 回 float16）
        return x.to(input_dtype)


class SwiGLU(nn.Module):
    """
    # 作业 3.4.2 SwiGLU激活函数
    SwiGLU 是激活函数，它通过将输入乘以sigmoid函数，然后乘以一个线性变换来得到输出。

    公式是：
    * out = w2(w1(x) * sigmoid(w1(x)) * w3(x))

    其中x是输入，w1(x)是线性变换，sigmoid(w1(x))是sigmoid函数，w2(x)是线性变换，w3(x)是线性变换。

    Args:
        d_model (int): 输入的维度
        d_ff (int): 输出的维度
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        # 直接用torch库中的Linear层
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # 升维到隐藏层
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # 降维回模型维度
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # 门控分支

    def forward(self, x):
        """ "
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            out: (batch_size, seq_len, d_model)
        """
        # SwiGLU: W2(SiLU(W1 x) ⊙ W3 x)
        # ?: self.w1() 是nn.Module 实现的 __call__() 方法（自动调用forward）
        return self.w2(self.silu(self.w1(x)) * self.w3(x))

    def silu(self, x):
        return x * torch.sigmoid(x)


class RoPE(nn.Module):
    """
    # 作业 3.4 RoPE位置编码
    Rotary Positional Embedding (RoPE): 通过旋转方式将位置信息注入 query 和 key 向量。

    公式：
        rope(x) = x[:, 2i]   * cos(pos * θ^(-2i/d_k))
                -- x[:, 2i+1] * sin(pos * θ^(-2i/d_k))

    支持任意批处理维度，并根据 token_positions 动态索引预计算的 cos/sin 缓存。

    Args:
        theta (float): 底数超参数
        d_k (int): 输入的维度，也就是d_model
        max_seq_len (int): 最大序列长度
        device (torch.device): 设备
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for RoPE")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # 计算频率基: 1 / θ^(2p/d_k)
        freqs = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, device=self.device).float() / d_k)
        )  # shape: [d_k//2]

        # 生成所有位置的旋转角度: [max_seq_len, d_k//2]
        positions = torch.arange(max_seq_len, device=self.device)
        sinusoids = torch.outer(positions, freqs)  # 外积，扩展为 [max_seq_len, d_k//2]

        # 缓存 cos 和 sin 编码，不参与训练
        self.register_buffer("cos_cache", sinusoids.cos(), persistent=False)
        self.register_buffer("sin_cache", sinusoids.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (*, seq_len, d_k), 输入张量（支持任意批维度）
            token_positions: shape (*, seq_len), 每个 token 的绝对位置索引

        Returns:
            shape (*, seq_len, d_k), 应用旋转编码后的输出
        """
        # 从缓存中取出对应位置的 cos 和 sin: shape (*, seq_len, d_k//2)
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        # 分割输入为偶数和奇数维度: x = [x_even, x_odd]
        x_even = x[..., 0::2]  # 偶数索引: 0, 2, 4, ...
        x_odd = x[..., 1::2]  # 奇数索引: 1, 3, 5, ...

        # 应用旋转公式(注意下标从0开始)
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        # 交错合并: 将 (even, odd) 沿最后一维堆叠并展平
        out = torch.stack([out_even, out_odd], dim=-1)  # shape: (*, seq_len, d_k//2, 2)
        out = out.flatten(-2)  # shape: (*, seq_len, d_k)

        return out


class CausalMultiHeadAttention(nn.Module):
    """
    # 作业 3.5 多头注意力机制

    因果多头自注意力模块，包含RoPE位置编码

    实现要点：
    1. 使用因果掩码确保每个位置只能关注之前的位置
    2. 将RoPE应用于查询(Q)和键(K)，但不应用于值(V)
    3. 按照Vaswani等人的设定，每个头的维度为 d_k = d_v = d_model/num_heads
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
    ):
        """
        初始化因果多头自注意力模块

        Args:
            d_model: 输入特征维度
            num_heads: 注意力头数量
            max_seq_len: 支持的最大序列长度
            theta: RoPE的底数超参数
            device: 设备
        """
        super().__init__()

        # 验证d_model可被num_heads整除
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.rope = RoPE(theta, self.head_dim, max_seq_len, device=self.device)
        # 这里要注意，我们使用多头注意力机制的时候，每个head的维度是d_model // n_heads，我们应当对每个head进行RoPE

        # 创建投影层（题目要求参数作为模块内部参数）
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        # 预先计算因果掩码（避免每次前向传播重复创建）
        causal_mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=self.device),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            token_positions: 每个token的绝对位置索引，形状为 (seq_len)
        Returns:
            输出张量，形状为 (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # 1. 投影到查询、键、值空间
        q: torch.Tensor = self.wq(x)  # (batch_size, seq_len, d_k)
        k: torch.Tensor = self.wk(x)  # (batch_size, seq_len, d_k)
        v: torch.Tensor = self.wv(x)  # (batch_size, seq_len, d_v)

        # 2. 重塑为多头格式: (batch_size, seq_len, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 转置
        # (batch_size, num_heads ,seq_len , head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # 3. 应用RoPE到查询和键（如果提供）
        # * 作业 3.5 优化版本
        if self.rope is not None:

            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # 4. 创建因果掩码
        mask: torch.Tensor = self.causal_mask[:seq_len, :seq_len]  # (seq_len, seq_len)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

        # 5. 计算缩放点积注意力
        # (batch_size, num_heads ,seq_len , head_dim)
        attention_output = self.attention(q, k, v, mask)

        # 6. 重塑回原始形状
        # contiguous确保内存连续
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.d_model)

        # 7. 应用输出投影
        output = self.wo(attention_output)

        return output

    def attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        """
        计算缩放点积注意力(**注意此处QKV已经完成多头注意力变换**)
        Args:
            Q: shape (batch_size, num_heads ,seq_len , head_dim)
            K: shape (batch_size, num_heads ,seq_len , head_dim)
            V: shape (batch_size, num_heads ,seq_len , head_dim)
            mask: shape (1, 1, seq_len, seq_len) 或 None
        Returns:
            shape (batch_size, num_heads ,seq_len , head_dim)
        """
        # d_k = self.head_dim
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))
        if mask is not None:
            # 如果mask为0，则将对应位置的score设置为-inf
            scores = scores.masked_fill(mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)  # 对key这一维度进行softmax归一化
        return torch.matmul(attn_weights, V)  # 将attn_weights与value相乘得到最终的输出


class TransformerBlock(nn.Module):
    """
    TransformerBlock 是Transformer块，它把包含多头注意力机制的一些组件包装在一起，形成一个完整的Transformer块。
    Args:
        d_model (int): 输入的维度，也就是d_model
        n_heads (int): 头的数量
        d_ff (int): 前馈神经网络的维度
        max_seq_len (int): 最大序列长度
        theta (float): 底数超参数
        attn_q_proj_weight (torch.Tensor): 查询的权重
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
    ):
        super(TransformerBlock, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.rms_norm1 = RMSNorm(d_model, eps=1e-5, device=device)
        self.rms_norm2 = RMSNorm(d_model, eps=1e-5, device=device)
        self.swiglu = SwiGLU(d_model, d_ff)

        self.causal_multi_head_attention = CausalMultiHeadAttention(
            d_model, n_heads, self.max_seq_len, self.theta, self.device
        )

    def forward(self, in_features: torch.Tensor):
        # 生成位置索引 (seq_len)
        token_positions = torch.arange(in_features.shape[1], device=in_features.device)
        x1 = self.rms_norm1(in_features)
        x1 = self.causal_multi_head_attention(x1, token_positions)
        x1 = x1 + in_features
        x2 = self.rms_norm2(x1)
        x2 = self.swiglu(x2)
        out = x2 + x1
        return out


class TransformerModule(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        n_layers: int,
        vocab_size: int,
        device: torch.device | None = None,
    ):
        super(TransformerModule, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.device = device

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, max_seq_len, theta, device)
                for _ in range(n_layers)
            ]
        )
        self.embedding_module = EmbeddingModule(vocab_size, d_model, device)
        self.linear_module = nn.Linear(d_model, vocab_size, bias=False, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding_module(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.linear_module(x)
        return x


class CosineSchedule:
    """
    实现带 warmup 的余弦退火学习率调度器
    学习率变化分为三个阶段：
    1. Warmup 阶段：从 0 线性上升到最大学习率
    2. Cosine 衰减阶段：从最大学习率按余弦函数下降到最小学习率
    3. 持平阶段：保持最小学习率不变
    """

    def __init__(
        self, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters
    ):
        """
        初始化余弦学习率调度器

        Args:
            max_learning_rate (float): 最大学习率（warmup 结束时达到的值）
            min_learning_rate (float): 最小学习率（余弦衰减结束时的值）
            warmup_iters (int): warmup 阶段的迭代次数
            cosine_cycle_iters (int): 余弦衰减周期的总迭代次数（从 warmup 结束到周期结束）
        """
        self.max_learning_rate = max_learning_rate  # 不修改大变量名
        self.min_learning_rate = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters

    def __call__(self, current_iter):
        """
        根据当前迭代次数计算对应的学习率

        Args:
            current_iter (int): 当前训练的迭代次数（从 0 开始）

        Returns:
            float: 该迭代次数对应的学习率值
        """
        # 阶段 1：Warmup 阶段（线性增长）
        if current_iter < self.warmup_iters:
            # 从 0 线性增长到 max_learning_rate
            return self.max_learning_rate * current_iter / self.warmup_iters

        # 阶段 3：Cosine 衰减结束后，保持最小学习率
        elif current_iter > self.cosine_cycle_iters:
            return self.min_learning_rate

        # 阶段 2：余弦退火衰减阶段
        else:
            # 计算从 warmup 结束到当前的相对位置（归一化到 [0, 1]）
            progress = (current_iter - self.warmup_iters) / (
                self.cosine_cycle_iters - self.warmup_iters
            )
            # 余弦衰减公式：从 1 到 0 的余弦曲线，再线性映射到 [min_lr, max_lr]
            cosine_decay = (1 + math.cos(math.pi * progress)) / 2
            return (
                self.min_learning_rate
                + (self.max_learning_rate - self.min_learning_rate) * cosine_decay
            )


class AdamW(optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """
        Simple AdamW implementation.
        """
        loss = None if closure is None else closure()

        for group in self.param_groups:
            p: nn.Parameter
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad  # use clean grad tensor
                state: dict[str, any] = self.state[p]

                # 初始化状态缓存：step, 变量 m, RMS v
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)  # 一阶矩估计
                    state["v"] = torch.zeros_like(p)  # 二阶矩估计

                m, v = state["m"], state["v"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # 更新变量：m = beta1 * m + (1 - beta1) * grad
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # 更新 RMS：v = beta2 * v + (1 - beta2) * grad^2
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 偏差修正系数
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] / bias_correction1

                # 分母：sqrt(v / bias_correction2) + eps
                denom = (v / bias_correction2).sqrt().add_(group["eps"])
                # 在 no_grad 环境中更新参数（避免 autograd 跟踪）
                with torch.no_grad():
                    p.addcdiv_(m, denom, value=-step_size)
                    # 解耦权重衰减（直接作用于参数，不通过梯度）
                    p.add_(p, alpha=-group["weight_decay"] * group["lr"])
        return loss


class CrossEntropyLoss:
    def __init__(self):
        """
        Args:
            inputs: [batch, seq, vocab]
            targets: [batch, seq]
        """
        pass

    def logsoftmax(self, inputs):
        inputs = inputs - torch.max(inputs, -1, keepdim=True)[0]  # 减去最大值，防止溢出
        # return torch.log(softmax(inputs,1))
        return inputs - torch.log(
            torch.sum(torch.exp(inputs), -1, keepdim=True)
        )  # logsumexp

    def nll_loss(self, inputs, targets):
        return torch.mean(
            -inputs[range(inputs.shape[0]), targets]
        )  # 花式索引，range(inputs.shape[0])是行索引，targets是列索引，找到对应位置的值也就是概率，这里使用的是one-hot编码，所以直接找到对应位置的值

    def forward(self, inputs, targets):
        batch, seq, vocab = inputs.shape
        # 先把原始输入展平为二维，然后计算log_softmax，最后计算nll_loss
        inputs = inputs.view(batch * seq, vocab)  # [batch*seq, vocab]
        targets = targets.view(batch * seq)  # [batch*seq]
        log_probs = self.logsoftmax(inputs)
        return self.nll_loss(log_probs, targets)
