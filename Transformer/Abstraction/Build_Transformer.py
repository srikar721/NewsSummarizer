import torch
import torch.nn as nn
import math

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, input_embed: InputEmbeddings, input_pos: PositionalEncoding, output_embed: InputEmbeddings, output_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_embed = input_embed
        self.output_embed = output_embed
        self.input_pos = input_pos
        self.output_pos = output_pos
        self.projection_layer = projection_layer

    def encode(self, input_seq, input_mask):
        input_seq = self.input_embed(input_seq)
        input_seq = self.input_pos(input_seq)
        return self.encoder(input_seq, input_mask)
    
    def decode(self, encoder_output: torch.Tensor, input_mask: torch.Tensor, output_seq: torch.Tensor, output_mask: torch.Tensor):
        output_seq = self.output_embed(output_seq)
        output_seq = self.output_pos(output_seq)
        return self.decoder(output_seq, encoder_output, input_mask, output_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(vocab_size: int, input_seq_len: int, output_seq_len: int, d_model: int=256, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=1024) -> Transformer:
    input_embed = InputEmbeddings(d_model, vocab_size)
    output_embed = InputEmbeddings(d_model, vocab_size)

    input_pos = PositionalEncoding(d_model, input_seq_len, dropout)
    output_pos = PositionalEncoding(d_model, output_seq_len, dropout)
    
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    projection_layer = ProjectionLayer(d_model, vocab_size)
    
    transformer = Transformer(encoder, decoder, input_embed, input_pos, output_embed, output_pos, projection_layer)
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer