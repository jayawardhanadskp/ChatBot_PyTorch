import nltk
nltk.download('punkt')

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer

def tokenize(sentaence):
    return nltk.word_tokenize(sentaence)

def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    pass