import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet

# Make sure the code runs only if this script is executed directly
if __name__ == "__main__":

    with open('intents.json', 'r') as f:
        intents = json.load(f)

    all_words = []
    tags = []
    xy = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)  # extend the list
            xy.append((w, tag))  # store the pattern and tag

    ignore_words = ['?', '!', '.', ',']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))
    print(tags)

    X_train = []
    y_train = []
    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)

        label = tags.index(tag)  # Fixed typo: 'lable' -> 'label'
        y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    class ChatDataset(Dataset):
        def __init__(self):
            self.n_samples = len(X_train)  # Fixed this line: no 'super()' needed
            self.x_data = X_train
            self.y_data = y_train

        def __getitem__(self, idx):
            return self.x_data[idx], self.y_data[idx]

        def __len__(self):
            return self.n_samples

    # Hyperparameters
    batch_size = 8
    hidden_size = 8
    output_size = len(tags)
    input_size = len(X_train[0])
    learning_rate = 0.001
    num_epochs = 1000

    # Dataset and DataLoader
    dataset = ChatDataset()
    # Set num_workers=0 to avoid multiprocessing issues
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()  # Fixed typo: 'critertion' -> 'criterion'
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:  # Fixed typo: 'lable' -> 'labels'
            words = words.to(device)
            labels = labels.to(device)

            # Forward pass
            output = model(words)  # Model output
            loss = criterion(output, labels)  # Compute loss

            # Backward pass and optimizer step
            optimizer.zero_grad()  # Zero the gradients
            loss.backward()  # Fixed typo: 'loss.backwaed()' -> 'loss.backward()'
            optimizer.step()  # Optimizer step

        if (epoch + 1) % 100 == 0:
            print(f'epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')

    print(f'final loss, loss={loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')