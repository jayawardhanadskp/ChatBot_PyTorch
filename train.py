import json
import numpy as np

from nltk_utils import tokenize, stem, bag_of_words

print(np.__version__)

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
        all_words.extend(w) # extend the list
        xy.append((w, tag)) # store the pattern and tag

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

    lable = tags.index(tag)
    y_train.append(lable)

X_train = np.array(X_train)
y_train = np.array(y_train)