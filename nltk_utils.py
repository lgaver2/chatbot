import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word)

#count 0 or 1 in fonction of all_words
def bag_of_words(tokenizd_sentence, all_words):
    tokenizd_sentence = [stem(w) for w in tokenizd_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    #give idx of word
    for idx,w in enumerate(all_words):
        if w in tokenizd_sentence:
            bag[idx] = 1.0
    return bag



