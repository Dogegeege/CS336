from config import *
from utilities import *
from BPETokenizer import BPETokenizer as Tokenizer
from train import find_latest_checkpoint


def top_p_sampling(probabilities, top_p=0.9):
    """
    Top-p 核采样
    """
    # 按概率降序排序
    sort_probabilities, idx = torch.sort(probabilities, dim=-1, descending=True)
    cumulative_probabilities = torch.cumsum(sort_probabilities, dim=-1)  # 累积概率

    threshold = top_p
    # 创建一个mask，用于将概率大于threshold的token设置为0，因为是降序排序，所以前面的token概率大
    mask = cumulative_probabilities > threshold

    # 将mask向右移动一个位置，确保起码有一个token的概率大于threshold
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = 0

    sort_probabilities[mask] = 0

    # 归一化
    sort_probabilities.div_(sort_probabilities.sum(dim=-1, keepdim=True))

    # 随机选择一个概率大于0的token
    next_token_idx = torch.multinomial(sort_probabilities, 1)
    next_token_idx = torch.gather(idx, dim=-1, index=next_token_idx)
    # 返回下一个token
    return next_token_idx


def temperature_scaling(logits, temperature=1.0):
    """
    温度缩放
    """
    probabilities = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
    return probabilities


def decode_token(
    input_tokens: torch.Tensor,
    model: nn.Module,
    max_tokens_to_generate: int,
    top_p=0.9,
    temperature=1.0,
):
    """
    解码推理.

    Args:
        input_tokens: 输入的token列表
        model: 训练好的模型
        max_tokens_to_generate: 最多生成的token数量
        top_p: top-p采样的概率阈值
        temperature: 温度缩放的温度值
    Returns:
        生成的token列表
    """
    model.eval()  # 设置为评估模式不要dropout

    input_tokens = input_tokens.clone().detach().unsqueeze(0)
    with torch.no_grad():  # 不计算梯度
        start_num_tokens = input_tokens.shape[1]
        for _ in range(start_num_tokens, max_tokens_to_generate):
            if input_tokens == "<endoftext>":
                break
            logits = model(input_tokens)
            probabilities = temperature_scaling(logits, temperature)
            next_token_idx = top_p_sampling(probabilities, top_p)

            # 将下一个token添加到input_ids中
            input_tokens = torch.cat([input_tokens, next_token_idx], dim=-1)

    return input_tokens


def predict():

    vocab_size = config["vocab_size"]
    context_length = config["context_length"]
    d_model = config["d_model"]
    d_ff = config["d_ff"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    rope_theta = config["rope_theta"]

    model = TransformerModule(
        d_model, n_heads, d_ff, context_length, rope_theta, n_layers, vocab_size, device
    ).to(device)
    find_latest_checkpoint(config.get("checkpoint_dir", "./checkpoints"))
    model.eval()

    tokenizer = Tokenizer(config)
    input_text = "Once upon a time, there was a pretty girl named Lily.One day, Lily’s mom asked her to help cook dinner."
    # input_text = "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of darkness, it was the spring of hope, it was the winter of despair."
    # input_text = "baby shark"

    input_ids = tokenizer.encode(input_text)
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)

    # 推理
    output_ids = decode_token(input_ids, model, max_tokens_to_generate=200)
    # print(output_ids)

    output_ids_list = output_ids[0].cpu().tolist()
    output_text = tokenizer.decode(output_ids_list)
    print(f"Input Text:\n {input_text}\n")

    print(f"Output Text:\n{output_text}")


if __name__ == "__main__":
    predict()
    # tokenizer = Tokenizer(config)
    # encode = tokenizer.encode("upon")
    # print(encode)
    # print(tokenizer.decode(encode))
