import json
from nltk_utils import tokenize, stem, bag_of_words

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
print(all_words)