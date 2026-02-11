import json
import os
import regex
from collections import defaultdict
from config import *
from typing import Dict, List, Tuple, Set, Iterable, Iterator
import base64

# GPT-2预分词模式
GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# 预编译正则以提高重复调用性能
WORD_RE = regex.compile(GPT2_SPLIT_PATTERN, flags=regex.UNICODE)


def gpt2_bytes_to_unicode_local() -> Dict[int, str]:
    """字节到Unicode映射（与训练时一致）"""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class BPETokenizer:
    def __init__(
        self, config: Dict = None, vocab_path: str = None, merges_path: str = None
    ):
        # 加载词汇表和合并规则
        if vocab_path is None:
            self.vocab = self._load_vocab(config["vocab_path"])
        else:
            self.vocab = self._load_vocab(vocab_path)
        if merges_path is None:
            self.merges = self._load_merges(config["merges_path"])
        else:
            self.merges = self._load_merges(merges_path)
        self.config = config
        if self.config == None:
            self.config = {
                "special_tokens": ["<|endoftext|>", "<pad>", "<unk>"],
            }

        # 创建反向映射(bytes -> ID)
        self.bytes_to_id = {bytes_val: idx for idx, bytes_val in self.vocab.items()}

        # *特殊token处理（更高效的查找）
        self.special_tokens = self.config["special_tokens"]
        self.special_to_id: Dict[str, int] = {}
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes in self.bytes_to_id:
                self.special_to_id[token] = self.bytes_to_id[token_bytes]

        # 创建合并优先级映射
        self.merges_priority_map = {pair: i for i, pair in enumerate(self.merges)}

        # 字节到Unicode映射（用于编码）
        self.bytes_to_unicode = gpt2_bytes_to_unicode_local()
        self.unicode_to_bytes = {v: k for k, v in self.bytes_to_unicode.items()}

        # 缓存：将常见单词的编码结果缓存为 token id 列表，显著加速大型文本编码
        self._encode_cache: Dict[bytes, List[int]] = {}

    def _load_vocab(self, path: str) -> Dict[int, bytes]:
        """加载词汇表文件"""
        with open(path, "r", encoding="utf-8") as f:
            vocab_str: dict = json.load(f)
        return {int(idx): token.encode("utf-8") for idx, token in vocab_str.items()}

    def _load_merges(self, path: str) -> List[Tuple[bytes, bytes]]:
        """加载合并规则文件"""
        merges = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
            parts = line.split()
            if len(parts) == 2:
                b64_t1, b64_t2 = parts
                t1 = base64.b64decode(b64_t1)
                t2 = base64.b64decode(b64_t2)
                merges.append((t1, t2))
        return merges

    def _bytes_to_unicode_str(self, byte_seq: bytes) -> str:
        """将字节序列转换为Unicode字符串（使用训练时的映射）"""
        return "".join(self.bytes_to_unicode[b] for b in byte_seq)

    def _unicode_str_to_bytes(self, unicode_str: str) -> bytes:
        """将Unicode字符串转换回字节序列"""
        return b"".join(bytes([self.unicode_to_bytes[c]]) for c in unicode_str)

    def _get_bpe_merges(self, piece: bytes) -> List[bytes]:
        """
        对字节片段进行BPE编码，返回字节列表
        """
        # 将字节转换为Unicode字符串（使用训练时的映射）
        unicode_str = self._bytes_to_unicode_str(piece)
        parts = [bytes([self.unicode_to_bytes[c]]) for c in unicode_str]

        while len(parts) > 1:
            # 查找所有可能的合并对
            pairs = set()
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                if pair in self.merges_priority_map:
                    pairs.add(pair)

            if not pairs:
                break

            # 找到优先级最高的合并对
            best_pair = min(pairs, key=lambda pair: self.merges_priority_map[pair])

            # 执行合并
            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best_pair:
                    new_parts.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts

        return parts

    def encode(self, text: str) -> List[int]:
        """将文本编码为token ID序列"""
        if not text:
            return []

        # 按特殊token分割文本
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        special_token_pattern = "|".join(map(regex.escape, sorted_special_tokens))

        if self.special_tokens:
            # 使用预编译正则进行分割/处理
            chunks = regex.split(f"({special_token_pattern})", text)
        else:
            chunks = [text]

        token_ids = []
        for chunk in chunks:
            if not chunk:
                continue

            if chunk in self.special_tokens:
                # 处理特殊token
                if chunk in self.special_to_id:
                    token_ids.append(self.special_to_id[chunk])
                else:
                    # 回退到UNK或第一个特殊token
                    if "<unk>" in self.special_to_id:
                        token_ids.append(self.special_to_id["<unk>"])
                    elif self.special_to_id:
                        token_ids.append(list(self.special_to_id.values())[0])
                    else:
                        token_ids.append(0)
            else:
                # 预分词（使用预编译正则）
                words: List[str] = WORD_RE.findall(chunk)
                for word in words:
                    if not word:
                        continue
                    # 获取单词的BPE tokens（并缓存最终的 token id 列表）
                    word_bytes = word.encode("utf-8")
                    if word_bytes in self._encode_cache:
                        token_ids.extend(self._encode_cache[word_bytes])
                        continue

                    pieces = self._get_bpe_merges(word_bytes)

                    # 将每个piece转换为token ID
                    piece_ids: List[int] = []
                    for piece in pieces:
                        if piece in self.bytes_to_id:
                            piece_ids.append(self.bytes_to_id[piece])
                        else:
                            # 处理未知token
                            if "<unk>" in self.special_to_id:
                                piece_ids.append(self.special_to_id["<unk>"])
                            elif self.special_to_id:
                                piece_ids.append(list(self.special_to_id.values())[0])
                            else:
                                piece_ids.append(0)

                    # 缓存并追加
                    self._encode_cache[word_bytes] = piece_ids
                    token_ids.extend(piece_ids)

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """将token ID序列解码为文本"""
        byte_sequence = b""
        for token_id in token_ids:
            if token_id in self.vocab:
                byte_sequence += self.vocab[token_id]
            else:
                # 处理无效token ID
                if "<unk>" in self.special_to_id:
                    unk_id = self.special_to_id["<unk>"]
                    if unk_id in self.vocab:
                        byte_sequence += self.vocab[unk_id]
                elif self.special_tokens:
                    # 使用第一个特殊token作为回退
                    first_special_id = list(self.special_to_id.values())[0]
                    if first_special_id in self.vocab:
                        byte_sequence += self.vocab[first_special_id]

        try:
            return byte_sequence.decode("utf-8", errors="replace")
        except UnicodeDecodeError:
            # 极端情况下的回退处理
            return byte_sequence.decode("latin1", errors="replace")

    def tokenize(self, text: str) -> List[str]:
        """将文本分词为token字符串（用于调试）"""
        token_ids = self.encode(text)
        tokens = []
        for token_id in token_ids:
            if token_id in self.vocab:
                try:
                    tokens.append(self.vocab[token_id].decode("utf-8"))
                except UnicodeDecodeError:
                    tokens.append(f"<BYTES:{self.vocab[token_id]}>")
            else:
                tokens.append("<INVALID_TOKEN>")
        return tokens


if __name__ == "__main__":
    # 配置路径
    output_dir = "./Experiments/data"
    vocab_path = os.path.join(output_dir, "gpt2_vocab.json")
    merges_path = os.path.join(output_dir, "gpt2_merges.txt")

    # 验证文件存在
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"词汇表文件不存在: {vocab_path}")
    if not os.path.exists(merges_path):
        raise FileNotFoundError(f"合并规则文件不存在: {merges_path}")

    print("🚀 加载训练好的分词器...")
    tokenizer = BPETokenizer(config, vocab_path, merges_path)
    print("✅ 分词器加载成功!")

    # 测试文本
    test_texts = [
        "Wow, that is great",
        "you can eat",
        "This is a test with special tokens: <|endoftext|>",
    ]

    # 添加特殊token测试
    test_texts.append(f"Special token test: {tokenizer.config['special_tokens'][0]}")

    print("\n🔍 开始分词器测试...")
    for text in test_texts:
        print(f"\n文本: {text}")

        # 编码
        token_ids = tokenizer.encode(text)
        print(
            f"编码 ({len(token_ids)} tokens): {token_ids[:20]}{'...' if len(token_ids) > 20 else ''}"
        )

        # 解码
        decoded_text = tokenizer.decode(token_ids)
        print(f"解码: {decoded_text}")

        # 验证往返一致性
        if text == decoded_text:
            print("✅ 往返一致")
        else:
            print("⚠️ 往返不一致")
            print(f"原始: {text}")
            print(f"解码: {decoded_text}")

            # 显示差异
            for i, (orig_char, dec_char) in enumerate(zip(text, decoded_text)):
                if orig_char != dec_char:
                    print(
                        f"位置 {i}: 原始 '{orig_char}' (U+{ord(orig_char):04X}) vs 解码 '{dec_char}' (U+{ord(dec_char):04X})"
                    )
                    break
            else:
                if len(text) != len(decoded_text):
                    print(f"长度不同: 原始 {len(text)} vs 解码 {len(decoded_text)}")

        # 显示前10个token
        tokens = tokenizer.tokenize(text)[:10]
        print(f"Token示例: {tokens}")

    print("\n✅ 测试完成!")
