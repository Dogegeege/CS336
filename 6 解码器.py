from training_loop import NanoTransformer
import torch


def top_p_sampling(probabilities: torch.Tensor, top_p: float = 0.9) -> torch.Tensor:
    """
    Top-p (nucleus) 采样

    Args:
        probabilities: 概率分布张量 (batch_size, vocab_size)
        top_p: 累计概率阈值 (0-1)

    Returns:
        next_token: 采样的下一个token索引 (batch_size, 1)
    """
    # 按概率降序排序
    sorted_probs, sorted_indices = torch.sort(probabilities, dim=-1, descending=True)

    # 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 创建移除mask: 累积概率超过top_p的位置
    mask = cumulative_probs > top_p

    # 保留第一个超过阈值的token，所以将mask右移一位
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False

    # 将需要移除的位置概率设为0
    sorted_probs[mask] = 0.0  # 选择mask=True的token概率为0

    # 归一化概率
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

    # 从处理后的分布中采样
    next_token_idx = torch.multinomial(sorted_probs, 1)

    # 获取原始词汇表中的token索引
    next_token = torch.gather(sorted_indices, dim=-1, index=next_token_idx)

    return next_token


def temperature_scaling(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    应用温度缩放到logits

    Args:
        logits: 模型输出的原始分数 (..., vocab_size)
        temperature: 温度参数(>0)

    Returns:
        probabilities: 缩放后的概率分布 (..., vocab_size)
    """
    # 应用温度缩放并计算概率
    scaled_logits = logits / temperature
    probabilities = torch.softmax(scaled_logits, dim=-1)
    return probabilities


def decode(
    model: torch.nn.Module,
    input_tokens: list,
    max_tokens_to_generate: int,
    context_length: int,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> list:
    """
    从语言模型生成文本补全

    Args:
        model: 训练好的语言模型
        input_tokens: 输入的token ID列表
        max_tokens_to_generate: 最大生成token数量
        context_length: 模型上下文长度
        temperature: 采样温度
        top_p: top-p采样阈值

    Returns:
        output_tokens: 包含输入和生成token的完整序列
    """
    model.eval()

    # 将输入转换为tensor并添加batch维度
    input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        # 复制输入作为当前序列
        current_sequence = input_tensor.clone()

        for _ in range(max_tokens_to_generate):
            # 如果序列过长，截取最后context_length个token
            if current_sequence.size(1) > context_length:
                model_input = current_sequence[:, -context_length:]
            else:
                model_input = current_sequence

            # 获取模型输出  (batch_size, context_length, vocab_size)
            logits = model(model_input)

            # 应用温度缩放获取概率分布 (batch_size, vocab_size)
            probs = temperature_scaling(logits[:, -1, :], temperature)

            # 使用top-p采样获取下一个token
            next_token = top_p_sampling(probs, top_p)

            # 将新token添加到序列
            current_sequence = torch.cat([current_sequence, next_token], dim=-1)

    # 移除batch维度并转换为列表
    return current_sequence.squeeze(0).tolist()


# 假设我们有一个训练好的模型和词汇表
model = NanoTransformer()  #!模型未加载训练权重
context_length = 16


# 初始输入token
input_tokens = [10, 20, 30]  # 示例token ID

# 生成补全
generated = decode(
    model=model,
    input_tokens=input_tokens,
    max_tokens_to_generate=10,
    context_length=context_length,
    temperature=0.8,
    top_p=0.9,
)

print("生成的token序列:", generated)
