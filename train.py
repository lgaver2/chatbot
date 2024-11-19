import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

#stemed words list
all_words = []
tags = []
#tuple word/tag
xy = []

# read the json file
for intent in intents['intents']:
    tag =  intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))


ignored_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignored_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))


x_train = []
y_train = []
for (pattern, tag) in xy:
    x_train.append(bag_of_words(pattern, all_words))
    y_train.append(tags.index(tag)) #get index of tag


x_train = np.array(x_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = torch.tensor(x_train, dtype=torch.float32)  # Inputs as FloatTensor
        self.y_data = torch.tensor(y_train, dtype=torch.long)     # Labels as LongTensor

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    
#params
batch_size = 8
input_size =  len(x_train[0])
output_size = len(tags)
hidden_size = 8
learning_rate = 0.001
num_epocs = 1000

dataset = ChatDataset()

train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = NeuralNet(input_size, hidden_size, output_size).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epocs):
    for (word, label) in train_loader:
        word = word.to(device)
        label = label.to(device)

        #forward
        outputs = model(word)
        loss = criterion(outputs, label)

        #back
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

