from BPEencoding import BPETokenizer as Tokenizer
from config import *
from tqdm import tqdm


def encodeAndSaveTokens(config, data_path):

    tokenizer = Tokenizer(config)

    # 加载训练数据
    with open(data_path, "r", encoding="utf-8") as f:
        original_data = f.read()

    chunk_size = 10000  # 每块字符数，可根据内存/速度调整
    encode_ids_list = []

    # 分块编码并显示进度条（避免一次性对超长文本编码）
    if len(original_data) <= chunk_size:
        # 如果文本较短，直接一次性编码
        encode_ids_list = tokenizer.encode(original_data)
    else:
        for i in tqdm(
            range(0, len(original_data), chunk_size), desc="🔄Encoding", unit="chunk"
        ):
            chunk = original_data[i : i + chunk_size]
            encode_ids_list.extend(tokenizer.encode(chunk))

    encode_ids = torch.tensor(encode_ids_list, dtype=torch.long)

    return encode_ids


# 编码验证数据

print("开始编码训练数据...")
data_path = config["train_data_path"]
train_encode_ids = encodeAndSaveTokens(config, data_path)
print(f"✅训练数据加载完成，tokens={len(train_encode_ids)}")

train_encode_ids_path = config["train_encode_ids_path"]
with open(train_encode_ids_path, "wb") as f:
    pickle.dump(train_encode_ids, f)
print(f"💾训练数据编码完成并已保存到{train_encode_ids_path}\n")


print("开始编码验证数据...")
data_path = config["valid_data_path"]
valid_encode_ids = encodeAndSaveTokens(config, data_path)
print(f"✅验证数据加载完成，tokens={len(valid_encode_ids)}")

valid_encode_ids_path = config["valid_encode_ids_path"]
with open(valid_encode_ids_path, "wb") as f:
    pickle.dump(valid_encode_ids, f)
print(f"💾验证数据编码完成并已保存到{valid_encode_ids_path}\n")
