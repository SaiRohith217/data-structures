import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
import matplotlib.pyplot as plt
import os

def train_bert(data):
    os.makedirs("../outputs", exist_ok=True)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2
)

    texts = list(data['text'])
    labels = list(data['label'])

    split = int(0.8 * len(texts))
    train_texts = texts[:split]
    train_labels = labels[:split]

    from src.transformer.dataset import SpamDataset

    train_dataset = SpamDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    losses = []

    model.train()

    for epoch in range(2):
        for batch in train_loader:
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

    # Save loss graph
    plt.plot(losses)
    plt.title("BERT Training Loss")
    plt.savefig("../outputs/bert_loss.png")
    plt.close()

    return model, tokenizer