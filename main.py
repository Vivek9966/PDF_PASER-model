import torch
import numpy as np
from torch  import nn
from torch.utils.data import DataLoader

class Inputembeddings(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.input_dim =input_dim

    def forward(self, x):
        return self.embedding(x)* (self.input_dim ** 0.5)
class Positionalembedding(nn.Module):
    def __init__(self, max_len, embedding_dim,dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.mat = torch.zeros(max_len, embedding_dim)
        self.position_vec = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        self.deno =  torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float)*np.log(10000)/embedding_dim)
        self.mat[:,0::2] = torch.sin(self.position_vec*self.deno)
        self.mat[:,1::2] = torch.cos(self.position_vec*self.deno)

        self.mat = self.mat.unsqueeze(0) # (1 ,max_len, embedding_dim)
        self.register_buffer('pe', self.mat)
    def forward(self, x):
        x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False)
        return self.dropout(x)
class Layernormalization(nn.Module):
    def __init__(self,eps = 10e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.gamma =  nn.Parameter(torch.zeros(1))
    def forward(self,x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = (x - mean) / (std + self.eps)
        return self.alpha * x + self.gamma
class Feedforward(nn.Module):
    def __init__(self, embedding_dim, dropout , intermediate_dimension):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, intermediate_dimension)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(intermediate_dimension, embedding_dim)
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x =self.linear2(x)

        return x
class MultiHead(nn.Module):
    def __init__(self, embedding_dim, num_heads,dropout):
        super().__init__()
        assert embedding_dim % num_heads == 0 , "Embedding dimension must be divisible by number of heads"
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.d_k = embedding_dim // num_heads
        self.w_q = nn.Linear(embedding_dim, embedding_dim)
        self.w_k = nn.Linear(embedding_dim, embedding_dim)
        self.w_v= nn.Linear(embedding_dim, embedding_dim)
        self.w_o = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.size(-1)
        scores = (query@ key.transpose(-2, -1)) / np.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0, 1e-8)
        scores = torch.softmax(scores, dim=-1) # batch h, seq,seq_len
        scores = dropout(scores)
        return (scores @ value), scores

    def forward(self , query, key, value,  mask=None):
        # Here you would implement the forward pass for multi-head attention
        # This is a placeholder for the actual implementation
        query = self.w_q(query)
        key = self.w_k(key)
        value = self.w_v(value)
        # (batch , sew, d_model )  => ( batch , selfnum hreads , sqe , dmodec)
        query = query.view(query.size(0), query.size(1), self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.size(0), key.size(1), self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.size(0), value.size(1), self.num_heads, self.d_k).transpose(1, 2)
        # (batch , num_heads, seq_len, d_k)
        x,self.scores = self.attention(query, key, value, mask, self.dropout)
        # (batch , num_heads, seq_len, d_k) => (batch, seq_len, num_heads, d_k)
        x = x.transpose(1,2).contiguous().view(x.size(0), -1, self.num_heads * self.d_k)
        # (batch, seq_len, num_heads * d_k)
        x = self.w_o(x)
        return x
class Residual(nn.Module):
    def __init__(self,dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = Layernormalization()
    def forward(self,x , sublayer):
        return self.norm(x + self.dropout(sublayer(x)))
class EncoderLayer(nn.Module):
    def __init__(self,self_attentition:MultiHead,feed_foreward:Feedforward, dropout):
        super().__init__()
        self.self_attention = self_attentition
        self.feed_forward = feed_foreward
        self.residual1 = nn.ModuleList([Residual(dropout), Residual(dropout)])
    def forward(self, x, src_mask=None):
        x = self.residual1[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.residual1[1](x, self.feed_forward)
        return x
class Encoder(nn.Module):
    def __init__(self, layers:nn.ModuleList, norm=None):
        super().__init__()
        self.layers = layers
        self.num_layers = len(layers)
        self.norm = Layernormalization()
    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        #if self.norm is not None:
        x = self.norm(x)
        return x
class DecoderLayer(nn.Module):
    def __init__(self, self_attentition:MultiHead,cross_attentition:MultiHead, feed_forward:Feedforward,  dropout):#layers:nn.ModuleList
        super().__init__()
        self.self_attention = self_attentition
        self.cross_attention = cross_attentition
        self.feed_forward = feed_forward
       # self.num_layers = len(layers)
        self.norm = Layernormalization()
        self.residual1 = nn.ModuleList([Residual(dropout), Residual(dropout), Residual(dropout)])
    def forward(self, x, output, tgt_mask=None, src_mask=None):
        x = self.residual1[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residual1[1](x, lambda x: self.cross_attention(x, output, output, src_mask))
        x = self.residual1[2](x, self.feed_forward)
        return x
class Decoder(nn.Module):
    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.num_layers = len(layers)
        self.norm = Layernormalization()
    def forward(self, x, output, tgt_mask=None, src_mask=None):
        for layer in self.layers:
            x = layer(x, output, tgt_mask, src_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x
class ProjectionLayer(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        return torch.softmax(self.linear(x),dim=-1)

class Transformer(nn.Module):
    def __init__(self, encoder:Encoder,decoder:Decoder,input_embedding:Inputembeddings,target_embedding:Inputembeddings, input_pos:Positionalembedding,target_pos:Positionalembedding, projection_layer:ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_embedding = input_embedding
        self.target_embedding = target_embedding
        self.input_pos = input_pos
        self.target_pos = target_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask=None):
        x = self.input_embedding(src)
        x = self.input_pos(x)
        return self.encoder(x, src_mask)
    def decode(self, encoder_output,src_mask,tgt,tgt_mask=None,):
        x = self.target_embedding(tgt)
        x = self.target_pos(x)
        return self.decoder(x, encoder_output, tgt_mask, src_mask)
    def project(self, x):
        return self.projection_layer(x)
def build_transformer(src_vocab_size:int,tgt_vocab_size,src_seq_len,tgt_seq_len:int,d_model = 512,N=10,h=8,dropout=.1,d_ff=2048):
    #create the input and target embeddings
    src_embedding = Inputembeddings(src_vocab_size, d_model)
    tgt_embedding = Inputembeddings(tgt_vocab_size, d_model)
    src_pos = Positionalembedding(src_seq_len, d_model, dropout)
    tgt_pos = Positionalembedding(tgt_seq_len, d_model, dropout)

    encoder_blocks =[]
    for i in range(N):
        self_attention = MultiHead(d_model, h, dropout)
        feed_forward = Feedforward(d_model, dropout, d_ff)
        encoder_block = EncoderLayer(self_attention, feed_forward, dropout)
        encoder_blocks.append(encoder_block)
    decoder_blocks = []
    for i in range(N):
        self_attention = MultiHead(d_model, h, dropout)
        cross_attention = MultiHead(d_model, h, dropout)
        feed_forward = Feedforward(d_model, dropout, d_ff)
        decoder_block = DecoderLayer(self_attention, cross_attention, feed_forward,  dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    # create aprojection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    #create transformer
    transformer = Transformer(encoder, decoder, src_embedding, tgt_embedding, src_pos, tgt_pos, projection_layer)

    # initiliaze the params
    for i in transformer.parameters():
        if i.dim() > 1:
            nn.init.xavier_uniform_(i)
        else:
            nn.init.constant_(i, 0)
    return transformer