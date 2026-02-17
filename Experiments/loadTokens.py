from BPETokenizer import BPETokenizer as Tokenizer
from BPEencoding import prepare_documents_from_dataset
from config import *
from tqdm import tqdm
import pickle
import torch
from datasets import load_dataset


def encodeAndSaveTokens(config, mode="train"):

    tokenizer = Tokenizer(config)

    # 加载 OpenWebText 数据集（若已本地缓存，load_dataset 会更快）
    print("🚀 加载 OpenWebText 数据集...")
    dataset = load_dataset("sytelus/openwebtext")

    # 准备训练文档
    original_data_list = prepare_documents_from_dataset(
        dataset,
        split=mode,
       sample_size= None, # None表示全部
        text_column="text",
    )

    encode_ids_list = []
    eos_token_id = tokenizer.config["special_tokens"][0]  # 假设tokenizer有该属性
    if eos_token_id==None:
        eos_token_id="<|endoftext|>"

    # 对每个文档编码，并添加eos标记
    for doc in tqdm(original_data_list, desc=f"Encoding {mode} data"):
        if not doc.strip():  # 跳过空文档
            continue
        ids = tokenizer.encode(doc)
        encode_ids_list.extend(ids)
        encode_ids_list.append(eos_token_id)  # 文档结束标记

    encode_ids = torch.tensor(encode_ids_list, dtype=torch.long)

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
