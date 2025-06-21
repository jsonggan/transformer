import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
  def __init__(self, d_model: int, vocab_size: int):
    super().__init()
    self.d_model = d_model
    self.vocab_size = vocab_size
    self.embedding = nn.Embedding(vocab_size, d_model)
    
  def forward(self, x):
    # Scaling by * math.sqrt(self.d_model) is used to maintain the variance of the embeddings at a suitable scale. Dot products can explode or vanish because the softmax function is sensitive to input scale.
    return self.embedding(x) * math.sqrt(self.d_model) 
  
  
class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
    super().__init()
    self.d_model = d_model
    self.seq_len = seq_len
    self.dropout = nn.Dropout(dropout)
    
    positional_encoding = torch.zeros(seq_len, d_model)
    
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    
    # use log() and exp() instead of just 10000 ** (some exponent) -> Exponentiation with large or small numbers can easily lead to overflow or underflow
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    positional_encoding[:, 0::2] = torch.sin(position * div_term)
    positional_encoding[:, 1::2] = torch.con(position * div_term)
    
    positional_encoding = positional_encoding.unsqueeze(0)
    
    self.register_buffer("pe", positional_encoding)
    
  def forward(self, x):
    x = x + (self.positional_encoding[:, :x.shape[1], :]).requires_grad_(False)
    return self.dropout(x)
  
class LayerNormalization(nn.Module):
  def __init__(self, eps: float = 10**-6) -> None:
    super().__init__()  
    self.eps = eps
    self.alpha = nn.Parameter(torch.ones(1))
    self.bias = nn.Parameter(torch.zeros(1))
    
  def forward(self, x):
    mean = x.mean(dim = -1, keepdim = True)
    std = x.std(dim = -1, keepdim = True)
    return self.alpha * (x - mean) / (std + self.eps) + self.bias
  
class FeedForwardBlock(nn.Module):
  def __init__(self, d_model: int, d_ff: int, dropout: int) -> None:
    super().__init__()
    self.linear_1 = nn.Linear(d_model, d_ff)
    self.dropout = nn.Dropout(dropout)
    self.linear_2 = nn.Linear(d_ff, d_model)
    
  def forward(self, x):
    return self.linear_2(self.dropout(torch.relu(self.linear_1(x)))) 
  
class MultiHeadAttentionBlock(nn.Module):
  def __init__(self, d_model: int, h: int, dropout: float) -> None:
    super().__init__()
    self.d_model = d_model
    self.h = h
    assert d_model % h == 0, "d_model is not divisible by h"
    
    self.d_k = d_model // h
    self.w_q = nn.Linear(d_model, d_model)
    self.w_k = nn.Linear(d_model, d_model)
    self.w_v = nn.Linear(d_model, d_model)
    
    self.w_o = nn.Linear(d_model, d_model)
    self.dropout = nn.Dropout(dropout)
  
  @staticmethod
  def attention(query, key, value, mask, dropout: nn.Dropout):
    d_k = query.shape[-1]
    
    attention_scores = (query @key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
      attention_scores.masked_fill_(mask == 0, -1e9)
    attention_scores = attention_scores.softmax(dim = -1)
    if dropout is not None:
      attention_scores = dropout(attention_scores)
    
    return (attention_scores @ value), attention_scores
  
  def forward(self, q, k, v, mask):
    # (batch_size, seq_len, d_model)
    query = self.w_q(q)
    key = self.w_k(k)
    value = self.w_v(v)
    
    # (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
    query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
    key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
    value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
    
    x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
    
    # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, d_model)
    # PyTorch may store transposed tensors in a non-contiguous memory layout, which makes .view() unsafe. .contiguous() ensures the tensor is stored in memory properly so .view() works as expected.
    x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
    
    return self.w_o(x)