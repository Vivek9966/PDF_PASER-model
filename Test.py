import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFKC
from pathlib import Path

import config
from Dataset import Bilingual_dataset,causal_mask
from  main import build_transformer
from torch.utils.tensorboard  import SummaryWriter
from config import get_weights_file_path
from tqdm import tqdm
import warnings
seqlen = 512
def get_all_sentences(ds, lang):

    for item in ds:
        yield item["translation"][lang]
def get_tokenizer(config,ds,lang):
    tokenizers_path = Path(config['tokenizer_file'],format(lang))
    if not Path.exists(tokenizers_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(

            special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'],min_frequency=2,
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizers_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizers_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizers_path))
    return tokenizer
def get_dataset(config):
    ds_raw = load_dataset( 'opus_books', 'en-fr', split='train')

    tokenizer_src = get_tokenizer(config, ds_raw, 'en')
    tokenizer_tgt = get_tokenizer(config, ds_raw, 'fr')

    #split
    train_ds_size = int(len(ds_raw) * 0.9)
    validation_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw , test_ds_raw = random_split(ds_raw , [train_ds_size, validation_ds_size])
    train_ds = Bilingual_dataset( train_ds_raw,tokenizer_src,tokenizer_tgt,seq_len=seqlen) #ds, tokenizer_src, tokenizer_tgt , seq_len,src='en', tgt='fr'
    val_ds = Bilingual_dataset( test_ds_raw,tokenizer_src,tokenizer_tgt,seq_len=seqlen)

    max_len_src =0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids =tokenizer_src.encode(item['translation']['en']).ids
        tgt_ids  = tokenizer_src.encode(item['translation']['fr']).ids
        max_len_src = max(max_len_src,len(src_ids))
        max_len_tgt = max(max_len_tgt,len(tgt_ids))
    print(f"Max len of src  = {max_len_src} \n Max Len of tgt {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], num_workers=4 ,shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
def get_model(config,vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len,vocab_tgt_len,512,512,) #src_vocab_size:int,tgt_vocab_size,src_seq_len,tgt_seq_len:int,d_model = 512,N=10,h=10,dropout=.1,d_ff=2048)
    return model
def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device{device}")
    Path(config['model_folder']).mkdir(parents=True,exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(config,tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size()).to(device)
     # Tensorboard
    writer =SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'],eps=1e-9)
    initial_epoch =0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config,config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch']+1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    loss_fn  =nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'),label_smoothing = 0.1).to(device)
    for epoch in range(initial_epoch,config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader,desc =f"Processing epoch {epoch:02d}" )
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device) # (B, seq_len)
            decoder_input = batch["decoder_input"].to(device) # (B, seq+)
            encoder_mask = batch["encoder_mask"].to(device) #'' ''
            decoder_mask = batch["decoder_mask"].to(device) # (b,1,seq,seq,len)

            # run tensore throufhg
            encoder_output = model.encode(encoder_input,encoder_mask)
            decoder_output = model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask)
            proje_op = model.projection_layer(decoder_output) # (b,s,tgtvs)

            label  = batch['label'].to(device) # b(seq)
            # b,s,tgtvs) --> b*s,tgtvs)
            loss  = loss_fn(proje_op.view(-1,tokenizer_tgt.get_vocab_size()),label.view(-1))

            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            #"log loss"
            writer.add_scalar('train_loss',loss.item(),global_step)
            writer.flush()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step+=1
            #save
            model_filename = get_weights_file_path(config,f'{epoch:02d}')
            torch.save({
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'global_step': global_step

            },model_filename)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = config.get_config()
    config['num_epochs'] = 30
    config['preload'] = None

    train_model(config)

