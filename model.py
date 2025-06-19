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