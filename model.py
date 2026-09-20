import torch
import torch.nn
import math

class InputEmbeddings(torch.nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(vocab_size, d_model)

    def forward(self, input_ids):
        return self.embedding(input_ids) * math.sqrt(self.d_model)

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model:int, max_len:int=512, dropout:float=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = torch.nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(torch.nn.Module):
    def __init__(self, eps:float=1e-6)->None:
        super().__init__()
        self.eps = eps
        self.alpha = torch.nn.Parameter(torch.ones(1)) # Multiplier parameter
        self.bias = torch.nn.Parameter(torch.zeros(1)) # Added parameter

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, keepdim=True, unbiased=False)
        return self.alpha * ((x - mean) / torch.sqrt(variance + self.eps)) + self.bias


class FeedForward(torch.nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float=0.1):
        super().__init__()
        self.linear1 = torch.nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        return self.linear2(self.dropout(torch.nn.functional.relu(self.linear1(x))))

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model:int, h:int, dropout:float=0.1):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        self.w_q = torch.nn.Linear(d_model, d_model)
        self.w_k = torch.nn.Linear(d_model, d_model)
        self.w_v = torch.nn.Linear(d_model, d_model)
        self.w_o = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout)    

    @staticmethod
    def attention(self, query, key, value, mask, dropout: torch.nn.Dropout):
        d_k = query.size(-1)

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return(attention_scores @ value), attention_scores
        

    def forward(self, query, key, value, mask=None):
        query = self.w_q(query)
        key = self.w_k(key)
        value = self.w_v(value)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(self, query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_k * self.h)

        return self.w_o(x)

class ResidualConnection(torch.nn.Module):
    def __init__(self, dropout:float):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(torch.nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual1 = ResidualConnection(dropout)
        self.residual2 = ResidualConnection(dropout)

    def forward(self, x, mask):
        x = self.residual1(x, lambda x: self.self_attention_block(x, x, x, mask))
        x = self.residual2(x, self.feed_forward_block)
        return x

class Encoder(torch.nn.Module):
    def __init__(self, layers: torch.nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(torch.nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual1 = ResidualConnection(dropout)
        self.residual2 = ResidualConnection(dropout)
        self.residual3 = ResidualConnection(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual1(x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual2(x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual3(x, self.feed_forward_block)
        return x


class Decoder(torch.nn.Module):
    def __init__(self, layers: torch.nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(torch.nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.linear = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.linear(x)


class Transformer(torch.nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, input_embeddings: InputEmbeddings, output_embeddings: InputEmbeddings, input_position: PositionalEncoding, output_position: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_embeddings = input_embeddings
        self.output_embeddings = output_embeddings
        self.input_position = input_position
        self.output_position = output_position
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src_embedded = self.input_embeddings(src)
        src_positioned = self.input_position(src_embedded)
        return self.encoder(src_positioned, src_mask)

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt_embedded = self.output_embeddings(tgt)
        tgt_positioned = self.output_position(tgt_embedded)
        return self.decoder(tgt_positioned, encoder_output, src_mask, tgt_mask)

    def project(self, decoder_output):
        return self.projection_layer(decoder_output)


def build_transformer(src_vocab_size:int, tgt_vocab_size:int, src_seq_len:int, tgt_seq_len:int, d_model:int = 512, num_layers:int=6, num_heads:int=8, d_ff:int=2048, dropout:float=0.1) -> Transformer:
    input_embeddings = InputEmbeddings(d_model, src_vocab_size)
    output_embeddings = InputEmbeddings(d_model, tgt_vocab_size)
    input_position = PositionalEncoding(d_model, src_seq_len, dropout=dropout)
    output_position = PositionalEncoding(d_model, tgt_seq_len, dropout=dropout)
    # projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    encoder_blocks = []
    for _ in range(num_layers):
        self_attention_block = MultiHeadAttention(d_model, num_heads, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(num_layers):
        self_attention_block = MultiHeadAttention(d_model, num_heads, dropout)
        cross_attention_block = MultiHeadAttention(d_model, num_heads, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(self_attention_block, cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(torch.nn.ModuleList(encoder_blocks))
    decoder = Decoder(torch.nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, input_embeddings, output_embeddings, input_position, output_position, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    return transformer
