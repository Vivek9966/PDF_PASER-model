import  torch
import torch.nn as nn
from   torch.utils.data import Dataset, DataLoader, random_split


#def causal_mask(param):
 #   pass


class Bilingual_dataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt , seq_len,src='en', tgt='fr'):
        self.src_lang = src
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.tgt_lang = tgt
        self.seq_len = seq_len
        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')],dtype=torch.long)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.long)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.long)
    def __len__(self):
        return len(self.ds)


    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding = self.seq_len - len(enc_input_tokens) -2
        dec_num_padding = self.seq_len - len(dec_input_tokens) -1

        if enc_num_padding < 0 or dec_num_padding < 0:
            raise ValueError("Sequence length is too short for the input data.")
        # Create encoder input tensor with SOS and EOS tokens
        encoder_input = torch.cat(
            [self.sos_token,
             torch.tensor(enc_input_tokens, dtype=torch.long),
             self.eos_token,
             torch.tensor([self.pad_token] * enc_num_padding, dtype=torch.long)],
        )
        # Create decoder input tensor with SOS token and EOS token
        decoder_input = torch.cat(
            [self.sos_token,
             torch.tensor(dec_input_tokens, dtype=torch.long),
             torch.tensor([self.pad_token] * dec_num_padding, dtype=torch.long)],
        )
        # Create decoder target tensor with EOS token
        assert encoder_input.shape[0] == self.seq_len, f"Encoder input shape mismatch: {encoder_input.shape[0]} != {self.seq_len}"
        assert decoder_input.shape[0] == self.seq_len, f"Decoder input shape mismatch: {decoder_input.shape[0]} != {self.seq_len}"
        assert decoder_input.shape[0] == self.seq_len, f"Decoder target shape mismatch: {decoder_input.shape[0]} != {self.seq_len}"

        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask':(encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.shape[0]),# (1, Seq_len) & (1,seqlen,seqlen)
            "label": decoder_input,
            'src_text':src_text,
            'tgt_txt' : tgt_text

        }
def causal_mask(decoder_input_mat):
        mask = torch.triu(torch.ones(1,decoder_input_mat,decoder_input_mat),diagonal=1).type(torch.int)
        return mask ==0