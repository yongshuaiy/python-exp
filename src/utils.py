import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def ensure_results_dir():
    """确保results目录存在"""
    os.makedirs('results', exist_ok=True)


def set_seed(seed=42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_training_curve(train_losses, val_losses, filename='training_curve.png'):
    """保存训练曲线图"""
    ensure_results_dir()
    save_path = os.path.join('results', filename)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Training curve saved to {save_path}")


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # 如果max_len为0，则不创建位置编码
        if max_len <= 0:
            self.register_buffer('pe', torch.zeros(0))
            self.no_encoding = True
            return

        self.no_encoding = False

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.no_encoding:
            return x  # 不添加位置编码
        return x + self.pe[:x.size(0), :]


def create_padding_mask(seq, pad_idx=0):
    """创建填充掩码"""
    # seq: (batch_size, seq_len)
    # 返回: (batch_size, 1, 1, seq_len)
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask.float()  # 转换为浮点数

def create_lookahead_mask(seq_len, device=None):
    """创建未来掩码（防止解码器看到未来信息）"""
    # 返回下三角矩阵，对角线及其以下为1，以上为0
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, seq_len)

def create_transformer_masks(src, tgt, pad_idx=0):
    """为完整Transformer创建所有需要的掩码"""
    device = src.device  # 获取输入张量的设备

    # 编码器掩码：仅填充掩码
    src_mask = create_padding_mask(src, pad_idx)

    # 解码器自注意力掩码：填充掩码 + 未来掩码
    tgt_padding_mask = create_padding_mask(tgt, pad_idx)
    lookahead_mask = create_lookahead_mask(tgt.size(1), device=device)

    # 确保两个掩码都是浮点数，然后相乘（相当于逻辑与）
    tgt_mask = tgt_padding_mask * lookahead_mask

    # 编码器-解码器注意力掩码：使用编码器的填充掩码
    memory_mask = src_mask

    return src_mask, tgt_mask, memory_mask

def create_lm_mask(seq, pad_idx=0):
    """为语言模型创建掩码（未来掩码 + 可选填充掩码）"""
    device = seq.device  # 获取输入张量的设备
    seq_len = seq.size(1)
    lookahead_mask = create_lookahead_mask(seq_len, device=device)

    # 如果有填充标记，添加填充掩码
    if pad_idx != 0:
        padding_mask = create_padding_mask(seq, pad_idx)
        return padding_mask * lookahead_mask  # 使用乘法而不是按位与
    else:
        return lookahead_mask