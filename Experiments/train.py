# 直接导入编码后的数据
from config import *
from utilities import *
from tqdm import tqdm

import pickle
import os
import glob
import random


def atomic_save(state, path):
    """原子化保存：先写临时文件再替换"""
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def find_latest_checkpoint(dir_path):
    """优先使用 final/latest.pth；否则按修改时间选择最近的 model_epoch_*.pth"""
    final = os.path.join(dir_path, "model_final_*.pth")
    final_files = glob.glob(final)
    if final_files:
        final_files.sort(key=os.path.getmtime, reverse=True)
        return final_files[0]

    latest = os.path.join(dir_path, "latest.pth")
    if os.path.exists(latest):
        return latest

    # 否则查找其他 checkpoint
    files = glob.glob(os.path.join(dir_path, "model_epoch_*.pth"))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def save_checkpoint(
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    lr_scheduler,
    path: str,
    timestamp=None,
):
    state = {
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "timestamp": timestamp or time.strftime("%Y%m%d_%H%M%S"),
        # 尝试保存 lr_scheduler 状态（如果支持）
        "lr_scheduler_state": (
            lr_scheduler.state_dict() if hasattr(lr_scheduler, "state_dict") else None
        ),
        # 保存随机状态以尽量保证可重复性
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "random_state": random.getstate(),
    }
    if torch.cuda.is_available():
        # 保存 CUDA 多卡的 rng 状态
        try:
            state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        except Exception:
            state["cuda_rng_state_all"] = None

    # 原子化写入目标路径，并更新 latest.pth
    checkpoint_dir = config.get("checkpoint_dir", "./checkpoints")
    atomic_save(state, path)
    latest_path = os.path.join(checkpoint_dir, "latest.pth")
    # 使用 replace 保持 atomic
    atomic_save(state, latest_path)


def load_checkpoint_if_exists(
    model: nn.Module, optimizer: optim.Optimizer, lr_scheduler
) -> dict | None:
    checkpoint_dir = config.get("checkpoint_dir", "./checkpoints")
    ckpt_path = find_latest_checkpoint(checkpoint_dir)
    if ckpt_path is None:
        return None

    try:
        # 将 numpy 的重构函数添加到安全白名单
        torch.serialization.add_safe_globals(
            [
                np._core.multiarray._reconstruct,
                np.ndarray,
                np.dtype,
                np.dtypes.UInt32DType,
            ]
        )
    except Exception:
        pass

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        # 如果 weights_only 失败且文件可信，回退到完整加载
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if ckpt.get("lr_scheduler_state") is not None and hasattr(
        lr_scheduler, "load_state_dict"
    ):
        try:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler_state"])
        except Exception:
            # 某些调度器可能无法完全恢复，这里容错
            pass
    # 恢复随机数状态
    if "torch_rng_state" in ckpt:
        try:
            torch.set_rng_state(ckpt["torch_rng_state"])
        except Exception:
            pass
    if (
        "cuda_rng_state_all" in ckpt
        and ckpt["cuda_rng_state_all"] is not None
        and torch.cuda.is_available()
    ):
        try:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state_all"])
        except Exception:
            pass
    if "numpy_rng_state" in ckpt:
        np.random.set_state(ckpt["numpy_rng_state"])
    if "random_state" in ckpt:
        random.setstate(ckpt["random_state"])
    return ckpt


def train():

    train_encode_ids_path = config.get("train_encode_ids_path")
    valid_encode_ids_path = config.get("valid_encode_ids_path")

    if not train_encode_ids_path:
        raise ValueError("❗配置中缺少 'train_encode_ids_path'")
    if not valid_encode_ids_path:
        raise ValueError("❗配置中缺少 'valid_encode_ids_path'")

    try:
        with open(train_encode_ids_path, "rb") as f:
            train_encode_ids = pickle.load(f)
        with open(valid_encode_ids_path, "rb") as f:
            valid_encode_ids = pickle.load(f)

    except (FileNotFoundError, PermissionError, ValueError, RuntimeError) as e:
        print(f"❌ 已分词 tokens 加载失败\n: {e}")
        # 根据你的需求决定是退出程序还是返回空数据
        import sys

        sys.exit(1)  # 退出程序
        # 或者设置空的数据加载器
        # train_data_loader = None
        # valid_data_loader = None
    train_data_loader = DataLoader(
        train_encode_ids, config["batch_size"], config["context_length"], shuffle=True
    )
    valid_data_loader = DataLoader(
        valid_encode_ids, config["batch_size"], config["context_length"], shuffle=True
    )

    # 加载模型
    model = TransformerModule(
        config["d_model"],
        config["n_heads"],
        config["d_ff"],
        config["context_length"],
        config["rope_theta"],
        config["n_layers"],
        config["vocab_size"],
        device,
    ).to(device)

    # 加载优化器和学习率调度器
    lr_scheduler = CosineSchedule(
        config["max_learning_rate"],
        config["min_learning_rate"],
        config["lr_warmup_steps"],
        config["cosine_cycle_iters"],
    )
    optimizer = AdamW(
        model.parameters(),
        config["initial_lr"],
        (config["adam_beta1"], config["adam_beta2"]),
        config["eps"],
        config["weight_decay"],
    )

    # 加载损失函数
    loss_fn = CrossEntropyLoss()

    print("✅模型加载完成\n")

    checkpoint_dir = config["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ==== 日志文件准备 ====
    log_dir = config["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    print(f"📅日志时间戳: {timestamp}")
    print(f"💻训练设备: {device}")
    print(f"验证间隔批次: {config['val_interval']} epochs")

    # 如果检测到已有 checkpoint，切换为恢复模式并将日志以追加模式打开
    ckpt = load_checkpoint_if_exists(model, optimizer, lr_scheduler)
    if ckpt is not None:
        resume = True
        resume_epoch = ckpt.get("epoch", 0)
        resume_step_in_epoch = ckpt.get("step_in_epoch", -1)
        global_step = ckpt.get("global_step", 0)
        started_timestamp = ckpt.get("timestamp", timestamp)
        log_mode = "a"  # append
        print(
            f"✔️从 checkpoint 恢复: epoch={resume_epoch}, step_in_epoch={resume_step_in_epoch}, global_step={global_step}"
        )
    else:
        resume = False
        resume_epoch = 0
        resume_step_in_epoch = -1
        global_step = 0
        started_timestamp = timestamp
        log_mode = "w"  # new log
        print("❌没有找到 checkpoint，开始新的训练")

    log_file_path = os.path.join(log_dir, f"training_log_{started_timestamp}.txt")
    log_file = open(log_file_path, log_mode, encoding="utf-8")
    log_file.write(
        f"✔️ Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}, resume={resume}\n"
    )
    log_file.flush()

    # 确保模型在正确设备
    model.to(device)

    # ==== 训练主循环（带恢复逻辑与中断保存） ====
    try:
        model.train()
        print("🚀开始训练...\n")
        for epoch in range(resume_epoch, config["epochs"]):
            # 如果 resume 时仍在同一个 epoch，需要从上次 step+1 开始
            if resume and epoch == resume_epoch:
                start_step = resume_step_in_epoch + 1
                # 如果上次 checkpoint 已经完成该 epoch（例如保存时 step_in_epoch = args.train_steps-1），则从0开始并且 resume=False
                if start_step >= getattr(
                    args, "train_steps", config.get("train_steps", 0)
                ):
                    start_step = 0
            else:
                start_step = 0

            with tqdm(
                range(start_step, args.train_steps),
                desc=f"🔄Epoch {epoch}",
                unit="step",
            ) as tbar:
                for step in tbar:
                    # 更新学习率
                    new_lr = lr_scheduler(global_step)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = new_lr

                    x, y = train_data_loader.get_train_batch_data()
                    x = x.to(device)
                    y = y.to(device)
                    logits = model(x)
                    loss = (
                        loss_fn(logits, y)
                        if callable(loss_fn)
                        else loss_fn.forward(logits, y)
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    global_step += 1

                    tbar.set_postfix(
                        {"loss": f"{loss.item():.6f}", "学习率": f"{new_lr:.6f}"}
                    )
                    tbar.update()

                    # 定期打印与写日志
                    if step % 100 == 0:
                        log_message = f"Epoch {epoch} Step {step} LR {new_lr:.6f} Loss: {loss.item():.6f} (global_step={global_step})"
                        log_file.write(log_message + "\n")
                        log_file.flush()

            # epoch 结束后写一次 epoch 完成日志
            log_message = f"Epoch {epoch} completed with loss: {loss.item():.6f}"
            print(log_message)
            print(f"💾日志已保存至📁 {log_file_path}\n")
            log_file.write(log_message + "\n")
            log_file.flush()

            # 保存周期性 checkpoint
            if (epoch + 1) % config["checkpoint_interval"] == 0:
                print(f"🛠️保存 checkpoint...")
                ckpt_name = os.path.join(
                    checkpoint_dir, f"model_epoch_{epoch}_{started_timestamp}.pth"
                )
                save_checkpoint(
                    epoch,
                    step,
                    global_step,
                    model,
                    optimizer,
                    lr_scheduler,
                    ckpt_name,
                    timestamp=started_timestamp,
                )
                print(f"💾Checkpoint 成功保存 epoch {epoch} 至文件📁 {ckpt_name}\n")
                log_file.write(
                    f"💾Checkpoint 成功保存 epoch {epoch} 至文件 {ckpt_name}\n"
                )
                log_file.flush()

            # 验证
            if (epoch + 1) % config["val_interval"] == 0:
                print(f"🔍开始验证...")
                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    val_steps = 0

                    with tqdm(
                        valid_data_loader.get_valid_batch_data_iter(),
                        desc="🔍验证",
                        unit="step",
                    ) as tbar:
                        for x_val, y_val in tbar:
                            x_val = x_val.to(device)
                            y_val = y_val.to(device)
                            logits = model(x_val)
                            loss_val = (
                                loss_fn(logits, y_val)
                                if callable(loss_fn)
                                else loss_fn.forward(logits, y_val)
                            )
                            val_loss += loss_val.item()
                            val_steps += 1
                        avg_val_loss = val_loss / max(1, val_steps)
                        log_message = (
                            f"验证 epoch {epoch}: 平均 loss: {avg_val_loss:.6f}"
                        )
                        log_file.write(log_message + "\n")
                        log_file.flush()

                        tbar.set_postfix({"平均验证 loss": f"{avg_val_loss:.6f}"})
                        tbar.update()
                    print(f"✅验证完成\n")
                model.train()

        # 训练完全结束，保存 final checkpoint
        final_ckpt = os.path.join(
            checkpoint_dir, f"model_final_{started_timestamp}.pth"
        )
        save_checkpoint(
            config["epochs"] - 1,
            args.train_steps - 1,
            global_step,
            model,
            optimizer,
            lr_scheduler,
            final_ckpt,
            timestamp=started_timestamp,
        )
        log_file.write("训练结束✅. Final checkpoint 已保存至: " + final_ckpt + "\n")
        print("训练结束✅. Final checkpoint 已保存至: ", final_ckpt)

    except KeyboardInterrupt:
        # 捕获 Ctrl+C 等中断，保存一个中断时的 checkpoint
        interrupt_ckpt = os.path.join(
            checkpoint_dir,
            f"interrupt_epoch_{epoch}_step_{step}_{started_timestamp}.pth",
        )
        save_checkpoint(
            epoch,
            step,
            global_step,
            model,
            optimizer,
            lr_scheduler,
            interrupt_ckpt,
            timestamp=started_timestamp,
        )
        msg = f"用户主动中断训练. Checkpoint 保存至📁 {interrupt_ckpt}\n"
        print(msg)
        log_file.write(msg)
        log_file.flush()
        raise  # 可选：重新抛出以便外部知晓中断

    except Exception as e:
        # 在发生未捕捉异常时也保存 checkpoint（有助于排查和恢复）
        error_ckpt = os.path.join(
            checkpoint_dir, f"error_epoch_{epoch}_step_{step}_{started_timestamp}.pth"
        )
        try:
            save_checkpoint(
                epoch,
                step,
                global_step,
                model,
                optimizer,
                lr_scheduler,
                error_ckpt,
                timestamp=started_timestamp,
            )
            log_file.write(
                f"Exception occurred: {e}. Checkpoint saved to {error_ckpt}\n"
            )
        except Exception as save_e:
            log_file.write(
                f"Exception occurred: {e}. Failed to save checkpoint: {save_e}\n"
            )
        log_file.flush()
        raise

    finally:
        log_file.write(f"Log closed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.flush()
        log_file.close()


if __name__ == "__main__":
    train()
