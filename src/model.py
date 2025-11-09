import math
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import PositionalEncoding

"""多头注意力机制"""
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        # 确保参数是整数
        d_model = int(d_model)
        num_heads = int(num_heads)
        dropout = float(dropout)

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        output = self.w_o(attn_output)
        return output


"""多头自注意力机制"""
class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, dropout)

    def forward(self, x, mask=None):
        return self.multi_head_attention(x, x, x, mask)


"""位置级FNN"""
class PositionWiseFFN(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1, activation="gelu"):
        super(PositionWiseFFN, self).__init__()
        # 确保参数是整数
        d_model = int(d_model)
        d_ff = int(d_ff)
        dropout = float(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


"""残差连接和层归一化"""
class ResidualConnection(nn.Module):

    def __init__(self, d_model, dropout=0.1):
        super(ResidualConnection, self).__init__()
        # 确保参数是整数
        d_model = int(d_model)
        dropout = float(dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 添加调试信息和类型检查
        if not callable(sublayer):
            print(f"ERROR: sublayer is not callable, type: {type(sublayer)}")
            print(f"sublayer value: {sublayer}")
            raise TypeError("sublayer must be callable")

        # 确保 sublayer 是一个函数或模块
        normalized_x = self.norm(x)
        sublayer_output = sublayer(normalized_x)
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, activation="gelu"):
        super(EncoderLayer, self).__init__()
        # 确保参数是整数
        d_model = int(d_model)
        num_heads = int(num_heads)
        d_ff = int(d_ff)
        dropout = float(dropout)

        self.self_attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFFN(d_model, d_ff, dropout, activation)
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)

    def forward(self, x, mask=None):
        x = self.residual1(x, lambda x: self.self_attention(x, mask))
        x = self.residual2(x, self.feed_forward)
        return x


"""Encoder编码器"""
class Encoder(nn.Module):

    def __init__(self, vocab_size, config):
        super(Encoder, self).__init__()
        model_config = config['model']

        d_model = int(model_config['d_model'])
        num_layers = int(model_config['num_layers'])
        num_heads = int(model_config['num_heads'])
        d_ff = int(model_config['d_ff'])
        max_seq_len = int(model_config['max_seq_len'])
        dropout = float(model_config['dropout'])
        activation = model_config['activation']

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # 嵌入和位置编码
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # 通过所有编码器层
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class DecoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, activation="gelu"):
        super(DecoderLayer, self).__init__()
        # 确保参数是整数
        d_model = int(d_model)
        num_heads = int(num_heads)
        d_ff = int(d_ff)
        dropout = float(dropout)

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFFN(d_model, d_ff, dropout, activation)

        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
        self.residual3 = ResidualConnection(d_model, dropout)

    # 交叉注意力：Q来自解码器，K和V来自编码器
    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        x = self.residual1(x, lambda x: self.self_attention(x, x, x, self_mask))
        x = self.residual2(x, lambda x: self.cross_attention(x, encoder_output, encoder_output, cross_mask))
        x = self.residual3(x, self.feed_forward)
        return x

"""Decoder——解码器"""
class Decoder(nn.Module):

    def __init__(self, vocab_size: int, config: Dict[str, Any]):
        super(Decoder, self).__init__()
        model_config = config['model']

        d_model = int(model_config['d_model'])
        num_layers = int(model_config['num_layers'])
        num_heads = int(model_config['num_heads'])
        d_ff = int(model_config['d_ff'])
        max_seq_len = int(model_config['max_seq_len'])
        dropout = float(model_config['dropout'])
        activation = model_config['activation']

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                self_mask: Optional[torch.Tensor] = None,
                cross_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 嵌入和位置编码
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # 通过所有解码器层
        for layer in self.layers:
            x = layer(x, encoder_output, self_mask, cross_mask)

        x = self.norm(x)
        return self.output_proj(x)


"""Transformer模型——"""
class TransformerLM(nn.Module):

    def __init__(self, vocab_size, config):
        super(TransformerLM, self).__init__()
        model_config = config['model']

        # 确保所有参数是正确类型
        d_model = int(model_config['d_model'])
        num_layers = int(model_config['num_layers'])
        num_heads = int(model_config['num_heads'])
        d_ff = int(model_config['d_ff'])
        max_seq_len = int(model_config['max_seq_len'])
        dropout = float(model_config['dropout'])
        activation = model_config['activation']

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 创建位置编码，如果max_seq_len为0，则不会添加位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.token_embedding(x)
        # 应用位置编码
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        return self.output_proj(x)

    def generate(self, input_ids, max_length=50, temperature=1.0, top_k=50, top_p=0.95):
        """
        自回归文本生成（仅解码器架构）
        """
        self.eval()
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # 前向传播
                output = self(generated)

                # 获取最后一个位置的logits
                next_token_logits = output[:, -1, :] / temperature

                # 应用采样策略
                next_token_id = self._sample_next_token(next_token_logits, top_k, top_p)

                # 添加到生成序列
                generated = torch.cat([generated, next_token_id.unsqueeze(1)], dim=1)

                # 简单的停止条件
                if (next_token_id == 0).all():  # 假设0是结束符
                    break

        return generated

    def _sample_next_token(self, logits, top_k, top_p):
        """采样下一个token"""
        # top-k过滤
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('Inf')

        # top-p过滤
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # 移除累积概率超过top_p的token
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')

        # 从剩余token中采样
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token


class Transformer(nn.Module):
    """完整的Transformer模型（编码器-解码器架构）"""

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, config: Dict[str, Any]):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, config)
        self.decoder = Decoder(tgt_vocab_size, config)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, memory_mask)
        return decoder_output

    def generate(self, src, max_length=50, temperature=1.0, top_k=50, top_p=0.95):
        """
        编码器-解码器文本生成
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device

        with torch.no_grad():
            # 编码器前向传播
            encoder_output = self.encoder(src)

            # 初始化目标序列（起始符）
            # 注意：这里假设0是起始符，您可能需要根据实际词汇表调整
            tgt = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

            for step in range(max_length - 1):
                # 创建目标掩码
                tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(device)

                # 解码器前向传播
                decoder_output = self.decoder(tgt, encoder_output, tgt_mask=tgt_mask)

                # 获取最后一个位置的logits
                next_token_logits = decoder_output[:, -1, :] / temperature

                # 应用采样
                next_token = self._sample_next_token(next_token_logits, top_k, top_p)

                # 添加到序列
                tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)

                # 停止条件（假设0是结束符）
                if (next_token == 0).all():
                    break

                if step % 10 == 0:
                    print(f"生成进度: {step + 1}/{max_length - 1}")

        return tgt

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """生成后续掩码，防止解码器看到未来信息"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _sample_next_token(self, logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
        """采样下一个token"""
        # top-k过滤
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('Inf')

        # top-p过滤
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # 移除累积概率超过top_p的token
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')

        # 从剩余token中采样
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token


"""创建模型"""


# def create_model(vocab_size, config):
#
#     model_type = config['model']['model_type']
#
#     if model_type == 'transformer_lm':
#         return TransformerLM(vocab_size, config)
#     else:
#         raise ValueError(f"Unknown model type: {model_type}")

def create_model(vocab_size: int, config: Dict[str, Any]) -> nn.Module:
    # model_type = config.get('model.model_type', 'transformer_lm')
    model_type = config['model']['model_type']
    print(model_type)

    if model_type == 'transformer_lm':
        return TransformerLM(vocab_size, config)
    elif model_type == 'transformer':
        #源和目标词汇表大小相同
        return Transformer(vocab_size, vocab_size, config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
