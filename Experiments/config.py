# 直接在代码中设置参数
import pickle
import time
import torch


class Args:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    epochs = 40
    train_steps = 5000
    batch_size = 32


args = Args()

device = args.device
epochs = args.epochs
train_steps = args.train_steps
batch_size = args.batch_size

timestamp = time.strftime("%Y%m%d_%H%M%S")

# 创建本地配置文件（代替 WandB）
config = {
    # 实验配置
    "experiment_name": f"tinystories_17M_{timestamp}",
    "total_tokens_processed": 9040017095,
    # 数据配置
    "train_data_path": "./data/TinyStoriesV2-GPT4-train.txt",
    "valid_data_path": "./data/TinyStoriesV2-GPT4-valid.txt",
    "vocab_path": "./Experiments/data/gpt2_vocab.json",
    "merges_path": "./Experiments/data/gpt2_merges.txt",
    "train_encode_ids_path": "./Experiments/data/encoded_ids_train.pkl",  # 训练数据编码后的保存路径
    "valid_encode_ids_path": "./Experiments/data/encoded_ids_valid.pkl",  # 验证数据编码后的保存路径
    "checkpoint_dir": "./Experiments/data/checkpoints",
    "data_dir": "./Experiments/data",
    "log_dir": "./Experiments/logs",
    # 编码器配置
    "special_tokens": ["<|endoftext|>", "<pad>", "<unk>"],
    # 模型配置
    "vocab_size": 50257,  # openwebtxt
    "context_length": 256,
    "d_model": 512,
    "d_ff": 1344,
    "n_layers": 4,
    "n_heads": 16,
    "rope_theta": 10000.0,
    # 训练配置
    "batch_size": batch_size,
    "initial_lr": 3e-5,
    "max_learning_rate": 3e-5,
    "min_learning_rate": 1e-5,
    "lr_warmup_steps": 2000,
    "cosine_cycle_iters": 10000,
    "weight_decay": 0.1,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "eps": 1e-8,
    "grad_clip": 1.0,
    "epochs": epochs,
    "train_steps": train_steps,
    # 日志和检查点配置
    "log_interval": 5,
    "val_interval": 40,
    "checkpoint_interval": 5,
}
