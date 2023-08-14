import torch
import torch.nn as nn
import math
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torchtext.utils import download_from_url, extract_archive
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
#from torchtext.legacy.data import Dataset, Example, Field
from torchtext.data import Dataset, Example, Field
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
#from torchtext.legacy.data import Dataset, Example, Field
from torchtext.data import Dataset, Example, Field
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchtext.data.utils import random_split
from torchtext.data.metrics import bleu_score, rouge_score
from transformer import build_transformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_newsroom_dataset():
    url = 'https://summarizationdatasets.blob.core.windows.net/newsroom/newsroom.tar.gz'
    raw_path = extract_archive(download_from_url(url))
    data_path = raw_path[0] / 'newsroom.tsv'

    # Load the dataset
    tokenizer = get_tokenizer('spacy')
    SRC = Field(tokenize=tokenizer, init_token='<sos>', eos_token='<eos>', lower=True)
    TGT = Field(tokenize=tokenizer, init_token='<sos>', eos_token='<eos>', lower=True)
    fields = [('src', SRC), ('tgt', TGT)]
    examples = []

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            src, tgt = line.strip().split('\t')
            examples.append(Example.fromlist([src, tgt], fields))

    dataset = Dataset(examples, fields)
    return dataset, SRC, TGT

def tokenize_dataset(dataset, src_field, tgt_field):
    src_field.build_vocab(dataset)
    tgt_field.build_vocab(dataset)
    return dataset

def get_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def validate_model(model, val_dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            src = batch.src.to(device)
            tgt = batch.tgt.to(device)

            src_mask = (src != model.src_embed.vocab.stoi['<pad>']).unsqueeze(-2)
            tgt_mask = (tgt != model.tgt_embed.vocab.stoi['<pad>']).unsqueeze(-2)

            tgt_inp = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            output = model.decode(model.encode(src, src_mask), src_mask, tgt_inp, tgt_mask)
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)

            tgt_out = tgt_out.contiguous().view(-1)

            loss = criterion(output, tgt_out)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_dataloader)
    return avg_loss

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, print_every=10):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for i, batch in enumerate(tqdm(train_dataloader)):
            src = batch.src.to(device)
            tgt = batch.tgt.to(device)

            src_mask = (src != model.src_embed.vocab.stoi['<pad>']).unsqueeze(-2)
            tgt_mask = (tgt != model.tgt_embed.vocab.stoi['<pad>']).unsqueeze(-2)

            tgt_inp = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            optimizer.zero_grad()

            output = model.decode(model.encode(src, src_mask), src_mask, tgt_inp, tgt_mask)
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)

            tgt_out = tgt_out.contiguous().view(-1)

            loss = criterion(output, tgt_out)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            total_loss += loss.item()

            if i % print_every == 0 and i > 0:
                avg_loss = total_loss / print_every
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_dataloader)}], Loss: {avg_loss:.4f}')
                total_loss = 0

        # Validation after every epoch
        val_loss = validate_model(model, val_dataloader, criterion)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}')

def main():
    # Parameters
    src_vocab_size = 100000
    tgt_vocab_size = 100000
    src_seq_len = 512
    tgt_seq_len = 512
    d_model = 256
    N = 6
    h = 6
    dropout = 0.1
    d_ff = 1024
    batch_size = 32
    num_epochs = 10

    dataset, SRC, TGT = prepare_newsroom_dataset()
    dataset = tokenize_dataset(dataset, SRC, TGT)

    # Create the Transformer model
    model = build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model, N, h, dropout, d_ff)

    # Initialize the optimizer and loss function
    optimizer = Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = get_dataloader(train_dataset, batch_size)
    val_dataloader = get_dataloader(val_dataset, batch_size)

    train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs)

    # Save
    #TODO Change to save model state after each epoch run
    torch.save(model.state_dict(), 'transformer_model.pt')

    # Perform inference 
    input_text = "Real-time summary"
    input_tokens = SRC.tokenize(input_text)
    input_indices = [SRC.vocab.stoi[token] for token in input_tokens]
    input_tensor = torch.tensor(input_indices).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        src_mask = (input_tensor != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
        memory = model.encode(input_tensor, src_mask)
        ys = torch.ones(1, 1).fill_(TGT.vocab.stoi['<sos>']).long().to(device)
        for i in range(100):
            tgt_mask = (ys != TGT.vocab.stoi['<pad>']).unsqueeze(-2)
            out = model.decode(memory, src_mask, ys, tgt_mask)
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim = 1)
            next_word = next_word.item()

            ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word).to(device)], dim=1)
            if next_word == TGT.vocab.stoi['<eos>']:
                break

        output_tokens = [TGT.vocab.itos[i] for i in ys.squeeze()]
        output_text = ' '.join(output_tokens[1:])  # Remove the initial '<sos>' token
        print("Generated Summary:", output_text)

if __name__ == "__main__":
    main()
