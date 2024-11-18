import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer

def tokenize(sentaence):
    return nltk.word_tokenize(sentaence)

def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    pass

a = "How long does delivery take?"
print(a)
a = tokenize(a)
print(a)