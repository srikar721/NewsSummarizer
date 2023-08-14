import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import numpy as np
from transformer import build_transformer
import spacy
from collections import Counter
from rouge import Rouge
import gzip
import json

def beam_search(model, input_tensor, max_length, beam_size):
    with torch.no_grad():
        model.eval()
        encoder_output = model.encode(input_tensor, None)
        initial_beam = [{'tokens': [vocab.word2index['<sos>']], 'log_prob': 0.0}]
        
        for _ in range(max_length):
            new_beam = []
            for candidate in initial_beam:
                last_token = candidate['tokens'][-1]
                if last_token == vocab.word2index['<eos>']:
                    new_beam.append(candidate)
                    continue
                
                input_tensor = torch.tensor([last_token]).to(device)
                output = model.decode(encoder_output, None, input_tensor, None)
                next_token_probs = nn.functional.log_softmax(output, dim=-1).squeeze()
                topk_probs, topk_indices = next_token_probs.topk(beam_size)
                
                for i in range(beam_size):
                    new_candidate = candidate.copy()
                    new_candidate['tokens'].append(topk_indices[i].item())
                    new_candidate['log_prob'] += topk_probs[i].item()
                    new_beam.append(new_candidate)
            
            new_beam.sort(key=lambda x: x['log_prob'], reverse=True)
            initial_beam = new_beam[:beam_size]
            
            if all(candidate['tokens'][-1] == vocab.word2index['<eos>'] for candidate in initial_beam):
                break
        
        return initial_beam[0]['tokens']

def generate_summary(model, input_text, beam_search_enabled=False, beam_size=5):
    model.eval()
    with torch.no_grad():
        input_indices = sequence_to_indices(input_text, vocab)
        input_tensor = torch.tensor(input_indices).unsqueeze(0).to(device)
        if beam_search_enabled:
            output_indices = beam_search(model, input_tensor, max_length=50, beam_size=beam_size)
        else:
            encoder_output = model.encode(input_tensor, None)
            output = model.decode(encoder_output, None, None, None)
            output_indices = output.argmax(dim=-1).squeeze().cpu().numpy()
        
        output_text = post_process_summary(output_indices)
    return output_text

def post_process_summary(summary):
    # Remove special tokens from the generated summary
    summary = [token for token in summary if token not in [vocab.word2index['<sos>'], vocab.word2index['<eos>']]]
    
    # Convert the output indices to text using the vocabulary
    output_text = " ".join([vocab.index2word[idx] for idx in summary])
    return output_text

def evaluate_rouge(predictions, targets):
    rouge = Rouge()
    scores = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}
    
    for pred, target in zip(predictions, targets):
        scores_ = rouge.get_scores(pred, target)[0]
        for metric in scores_:
            scores[metric].append(scores_[metric]['f'])
    
    avg_scores = {metric: sum(scores[metric]) / len(scores[metric]) for metric in scores}
    return avg_scores

def validate_and_evaluate(model, val_loader, beam_search_enabled=False, beam_size=5):
    model.eval()
    with torch.no_grad():
        predictions = []
        targets = []
        for article, summary in tqdm(val_loader):
            article = article.to(device)
            summary = summary.to(device)
            encoder_output = model.encode(article, None)
            if beam_search_enabled:
                output_indices = beam_search(model, article, max_length=50, beam_size=beam_size)
            else:
                output = model.decode(encoder_output, None, None, None)
                output_indices = output.argmax(dim=-1).squeeze().cpu().numpy()
            output_text = post_process_summary(output_indices)
            predictions.append(tokenize(output_text))
            targets.append([vocab.index2word[idx] for idx in summary.cpu().numpy()])
        
        rouge_scores = evaluate_rouge(predictions, targets)
        return rouge_scores

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def load_jsonl_gz(file_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

# Load train, test, and validation data
train_data = load_jsonl_gz('train.jsonl.gz')
test_data = load_jsonl_gz('test.jsonl.gz')
val_data = load_jsonl_gz('dev.jsonl.gz')

articles_train = [item['text'] for item in train_data]
summaries_train = [item['summary'] for item in train_data]

articles_test = [item['text'] for item in test_data]
summaries_test = [item['summary'] for item in test_data]

articles_val = [item['text'] for item in val_data]
summaries_val = [item['summary'] for item in val_data]


nlp = spacy.load("en_core_web_sm")

def tokenize(text):
    return [token.text for token in nlp(text)]

articles_train = [tokenize(article) for article in articles_train]
summaries_train = [tokenize(summary) for summary in summaries_train]
articles_val = [tokenize(article) for article in articles_val]
summaries_val = [tokenize(summary) for summary in summaries_val]

class Vocabulary:
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.word2count = {}
        self.num_words = 0

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

vocab = Vocabulary()
for article, summary in zip(articles_train, summaries_train):
    vocab.add_sentence(" ".join(article))
    vocab.add_sentence(" ".join(summary))

def sequence_to_indices(sequence, vocab):
    return [vocab.word2index[word] for word in sequence]

articles_train = [sequence_to_indices(article, vocab) for article in articles_train]
summaries_train = [sequence_to_indices(summary, vocab) for summary in summaries_train]
articles_val = [sequence_to_indices(article, vocab) for article in articles_val]
summaries_val = [sequence_to_indices(summary, vocab) for summary in summaries_val]

vocab_size = vocab.num_words
seq_len = 512  # Specify the maximum sequence length for padding/truncation
d_model = 256
N = 6
h = 8
dropout = 0.1
d_ff = 1024

# Initialize the Transformer model using the build_transformer function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_transformer(vocab_size, seq_len, d_model, N, h, dropout, d_ff).to(device)
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define a custom dataset class and data loader for validation
class CustomDataset(Dataset):
    def __init__(self, articles, summaries):
        self.articles = articles
        self.summaries = summaries

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        return self.articles[idx], self.summaries[idx]

train_dataset = CustomDataset(articles_train, summaries_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = CustomDataset(articles_val, summaries_val)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Step 4: Model Training with Validation
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for article, summary in tqdm(train_loader):
        article = article.to(device)
        summary = summary.to(device)
        optimizer.zero_grad()
        encoder_output = model.encode(article, None)
        output = model.decode(encoder_output, None, summary, None)
        loss = criterion(output, summary)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}")

    # Validation
    val_scores = validate_and_evaluate(model, val_loader, beam_search_enabled=True, beam_size=5)
    print(f"Validation ROUGE Scores: {val_scores}")

test_dataset = CustomDataset(articles_test, summaries_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Final Evaluation on Test Set
test_scores = validate_and_evaluate(model, test_loader)
print(f"Test ROUGE Scores: {test_scores}")

input_text = "This is an example input article."
generated_summary = generate_summary(model, input_text, beam_search_enabled=True, beam_size=5)
print("Generated Summary with Beam Search:", generated_summary)

# Evaluate the generated summary using ROUGE scores
rouge_scores_beam = validate_and_evaluate(model, val_loader, beam_search_enabled=True, beam_size=5)
print("Validation ROUGE Scores with Beam Search:", rouge_scores_beam)