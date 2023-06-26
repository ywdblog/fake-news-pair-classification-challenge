import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import jieba.posseg as pseg
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
 

class TextClassificationModel(nn.Module):
    def __init__(self, num_words, embedding_dim, lstm_units, num_classes):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(num_words, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_units)
        #self.fc = nn.Linear(lstm_units, num_classes)
        self.fc = nn.Linear(2 * lstm_units, num_classes)

    def forward(self, x1, x2):
 
        embedded1 = self.embedding(x1)
        embedded2 = self.embedding(x2)
        lstm_output1, _ = self.lstm(embedded1)
        lstm_output2, _ = self.lstm(embedded2)
        lstm_output1 = lstm_output1[:, -1, :]
        lstm_output2 = lstm_output2[:, -1, :]
        concatenated = torch.cat((lstm_output1, lstm_output2), dim=1)
        logits = self.fc(concatenated)
        return logits
  



# Tokenize text using jieba
def jieba_tokenizer(text):
    words = pseg.cut(text)
    return [word for word, flag in words if flag != 'x']

# Custom dataset class
class TextDataset(Dataset):
    def __init__(self, x_train1, x_train2, y_train):
        self.x_train1 = x_train1
        self.x_train2 = x_train2
        self.y_train = y_train

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        x1 = self.x_train1[idx]
        x2 = self.x_train2[idx]
        y = self.y_train[idx]
        return x1, x2, y

# Set random seed for reproducibility
torch.manual_seed(42)

TRAIN_CSV_PATH = "train-small.csv"
TEST_CSV_PATH = "test-small.csv"

# Load and preprocess the data
train_data = pd.read_csv(TRAIN_CSV_PATH, index_col=0)
train_data = train_data[['title1_zh', 'title2_zh', 'label']]
train_data['title1_tokenized'] = train_data['title1_zh'].apply(jieba_tokenizer)
train_data['title2_tokenized'] = train_data['title2_zh'].apply(jieba_tokenizer)
train_labels = train_data['label'].map({'unrelated': 0, 'agreed': 1, 'disagreed': 2}).values

tokenizer = {}
tokenizer['word_index'] = {}
tokenizer['index_word'] = {}

# Tokenize the text and build the vocabulary
def tokenize_text(text):
    tokens = []
    for word in text:
        if word not in tokenizer['word_index']:
            index = len(tokenizer['word_index']) + 1
            tokenizer['word_index'][word] = index
            tokenizer['index_word'][index] = word
        tokens.append(tokenizer['word_index'][word])
    return tokens

 

train_data['title1_tokenized'] = train_data['title1_tokenized'].apply(tokenize_text)
train_data['title2_tokenized'] = train_data['title2_tokenized'].apply(tokenize_text)

# Pad sequences to the same length
MAX_SEQUENCE_LENGTH = 20
train_data['title1_padded'] = train_data['title1_tokenized'].apply(lambda x: x[:MAX_SEQUENCE_LENGTH] + [0] * (MAX_SEQUENCE_LENGTH - len(x)))
train_data['title2_padded'] = train_data['title2_tokenized'].apply(lambda x: x[:MAX_SEQUENCE_LENGTH] + [0] * (MAX_SEQUENCE_LENGTH - len(x)))

# 
train_data['title1_padded'] = train_data['title1_padded'].apply(lambda x: [int(i) for i in x])
train_data['title2_padded'] = train_data['title2_padded'].apply(lambda x: [int(i) for i in x])


x_train1, x_val1, x_train2, x_val2, y_train, y_val = train_test_split(
    train_data['title1_padded'].values,
    train_data['title2_padded'].values,
    train_labels,
    test_size=0.1,
    random_state=42
)
 

# print(x_train1)
# import sys 
# sys.exit(0)

# Convert the data to PyTorch tensors
# Convert the data to PyTorch tensors
# x_train1 = torch.LongTensor(x_train1)
# x_train2 = torch.LongTensor(x_train2)
# y_train = torch.LongTensor(y_train)
# x_val1 = torch.LongTensor(x_val1)
# x_val2 = torch.LongTensor(x_val2)
# y_val = torch.LongTensor(y_val)


x_train1 = torch.LongTensor(x_train1.tolist())
x_train2 = torch.LongTensor(x_train2.tolist())
y_train = torch.LongTensor(y_train.tolist())
x_val1 = torch.LongTensor(x_val1.tolist())
x_val2 = torch.LongTensor(x_val2.tolist())
y_val = torch.LongTensor(y_val.tolist())

 
# Define the hyperparameters
NUM_WORDS = len(tokenizer['word_index']) + 1
EMBEDDING_DIM = 256
LSTM_UNITS = 128
NUM_CLASSES = 3
BATCH_SIZE = 512
NUM_EPOCHS = 1

# Create the data loaders
train_dataset = TextDataset(x_train1, x_train2, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = TextDataset(x_val1, x_val2, y_val)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Create the model
model = TextClassificationModel(NUM_WORDS, EMBEDDING_DIM, LSTM_UNITS, NUM_CLASSES)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters())
import sys

 
if sys.argv[1] == 'train':
    # Train the model
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for inputs1, inputs2, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs1, inputs2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs1.size(0)
            _, predicted = torch.max(outputs, 1)
            train_acc += (predicted == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        #.no_grad() 表示在这个代码块中不计算梯度，也不进行反向传播，原因是我们在验证集上进行前向传播时不需要计算梯度，也不需要进行反向传播
        with torch.no_grad():
            for inputs1, inputs2, labels in val_loader:
                outputs = model(inputs1, inputs2)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs1.size(0)
                _, predicted = torch.max(outputs, 1)
                val_acc += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print()
    torch.save(model.state_dict(), 'model.pt')
else :
    # Load the model
    model = TextClassificationModel(NUM_WORDS, EMBEDDING_DIM, LSTM_UNITS, NUM_CLASSES)
    model.load_state_dict(torch.load('model.pt'))
    model.eval()


import jieba.posseg as pseg
import torch

# Tokenize text using jieba
def jieba_tokenizer(text):
    words = pseg.cut(text)
    return [word for word, flag in words if flag != 'x']

# Tokenize the text and build the vocabulary
def tokenize_text(text, tokenizer):
    tokens = []
    for word in text:
        if word not in tokenizer['word_index']:
            continue  # Ignore words not in the tokenizer's vocabulary
        tokens.append(tokenizer['word_index'][word])
    return tokens

tokenizer = {'word_index': {}, 'index_word': {}}

# Load and preprocess the test data
test_data = pd.read_csv(TEST_CSV_PATH, index_col=0)
test_data = test_data[['title1_zh', 'title2_zh']]
test_data['title1_tokenized'] = test_data['title1_zh'].apply(jieba_tokenizer)
test_data['title2_tokenized'] = test_data['title2_zh'].apply(jieba_tokenizer)

test_data['title1_tokenized'] = test_data['title1_tokenized'].apply(lambda x: tokenize_text(x, tokenizer))
test_data['title2_tokenized'] = test_data['title2_tokenized'].apply(lambda x: tokenize_text(x, tokenizer))

test_data['title1_padded'] = test_data['title1_tokenized'].apply(lambda x: x[:MAX_SEQUENCE_LENGTH] + [0] * (MAX_SEQUENCE_LENGTH - len(x)))
test_data['title2_padded'] = test_data['title2_tokenized'].apply(lambda x: x[:MAX_SEQUENCE_LENGTH] + [0] * (MAX_SEQUENCE_LENGTH - len(x)))

x_test1 = torch.LongTensor(test_data['title1_padded'].tolist())
x_test2 = torch.LongTensor(test_data['title2_padded'].tolist())

class TestDataset2(Dataset):
    def __init__(self, x_test1, x_test2):
        self.x_test1 = x_test1
        self.x_test2 = x_test2

    def __len__(self):
        return len(self.x_test1)

    def __getitem__(self, idx):
        x1 = self.x_test1[idx]
        x2 = self.x_test2[idx]
        return x1, x2
    
# Create the test dataset and data loader
test_dataset = TestDataset2(x_test1, x_test2)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Make predictions
predictions = []
with torch.no_grad():
    for inputs1, inputs2 in test_loader:
        outputs = model(inputs1, inputs2)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.tolist())

# Map predicted labels back to original labels
label_map = {0: 'unrelated', 1: 'agreed', 2: 'disagreed'}
predicted_labels = [label_map[prediction] for prediction in predictions]

# Print the predicted labels

print(predicted_labels)
