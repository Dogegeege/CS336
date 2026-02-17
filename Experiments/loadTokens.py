from BPETokenizer import BPETokenizer as Tokenizer
from BPEencoding import prepare_documents_from_dataset
from config import *
from tqdm import tqdm
import pickle
import torch
from datasets import load_dataset


def encodeAndSaveTokens(config, mode="train"):
    tokenizer = Tokenizer(config)

    # 从配置中获取样本大小，若无则使用默认值（例如 10000）
    sample_size = config.get(f"{mode}_sample_size", 10000)

    print(f"🚀 加载 OpenWebText 数据集作为 {mode} 数据（前 {sample_size} 篇）...")

    # 加载数据集时直接切片，避免下载全部
    if mode == "train":
        # 训练集：从头开始取 sample_size 篇
        dataset = load_dataset(
            "sytelus/openwebtext",
            split=f"train[:{sample_size}]"
        )
        print(f"✅ 训练集加载完成，共 {len(dataset)} 篇文档")
    else:
        # 验证集：从训练集末尾之后开始取 sample_size 篇
        # 需要知道训练集的大小，以便偏移起始索引
        train_size = config.get("valid_sample_size", 1000)
        start = train_size
        end = start + sample_size
        dataset = load_dataset(
            "sytelus/openwebtext",
            split=f"train[{start}:{end}]"
        )
        print(f"✅ 验证集加载完成，共 {len(dataset)} 篇文档")

    # 准备文档列表（假设 prepare_documents_from_dataset 能直接处理 dataset 并提取 text 列）
    original_data_list = prepare_documents_from_dataset(
        dataset,
        split=mode,
        sample_size=None,  # 因为 dataset 已经切片，此处无需再限制
        text_column="text",
    )

    # 获取 eos_token_id（推荐直接从 tokenizer 对象获取）
    eos_token_id = tokenizer.special_to_id["<|endoftext|>"]  # 根据您的 Tokenizer 实现调整
    # 如果上述方法不可用，可硬编码：eos_token_id = 50256（GPT-2 的 eos 值）

    encode_ids_list = []

    # 逐个文档编码，添加 eos 标记
    for doc in tqdm(original_data_list, desc=f"Encoding {mode} data"):
        if not doc.strip():
            continue
        ids = tokenizer.encode(doc)
        encode_ids_list.extend(ids)
        encode_ids_list.append(eos_token_id)

    encode_ids = torch.tensor(encode_ids_list, dtype=torch.long)

    # 可选：保存到配置指定的路径
    save_path = config.get(f"{mode}_encode_ids_path")
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(encode_ids, f)
        print(f"✅ 已保存编码后的 token ids 到 {save_path}")

    return encode_ids


# 编码验证数据

print("开始编码训练数据...")
train_encode_ids = encodeAndSaveTokens(config,mode="train")
print(f"✅训练数据加载完成，tokens={len(train_encode_ids)}")

train_encode_ids_path = config["train_encode_ids_path"]
with open(train_encode_ids_path, "wb") as f:
    pickle.dump(train_encode_ids, f)
print(f"💾训练数据编码完成并已保存到{train_encode_ids_path}\n")


print("开始编码验证数据...")
valid_encode_ids = encodeAndSaveTokens(config,mode="valid")
print(f"✅验证数据加载完成，tokens={len(valid_encode_ids)}")

valid_encode_ids_path = config["valid_encode_ids_path"]
with open(valid_encode_ids_path, "wb") as f:
    pickle.dump(valid_encode_ids, f)
print(f"💾验证数据编码完成并已保存到{valid_encode_ids_path}\n")
