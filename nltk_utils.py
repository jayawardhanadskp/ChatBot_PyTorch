import nltk
import numpy as np

# nltk.download('punkt')  # Uncomment this line if you haven't downloaded the punkt tokenizer

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """
    Return a bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise.
    
    Example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag = [0, 1, 0, 1, 0, 0, 0]
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    # Create a bag of words (vector) with 0s for each word that is not in the sentence
    bag = np.zeros(len(all_words), dtype='float32')

    # If the word exists in the tokenized sentence, mark it with a 1 in the corresponding index
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag

# sentence = ["hello", "how", "are", "you"]
# words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
# bog = bag_of_words(sentence, words)
# print(bog)

# # Sample sentence and all_words list
# sentence = "Hello, how are you?"
# tokenized_sentence = tokenize(sentence)
# words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]

# # Get the bag of words
# bog = bag_of_words(tokenized_sentence, words)

# print("Bag of words:", bog)





# a = "How long does delivery take?"
# print(a)
# a = tokenize(a)
# print(a)

# words = ["Organize", "organizes", "organizing"]
# stemmed_words = [stem(w) for w in words]
# print(stemmed_words)  # Should output: ['organ', 'organ', 'organ']