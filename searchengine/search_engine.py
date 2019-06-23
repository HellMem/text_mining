import matplotlib.pyplot as plt
import matplotlib
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk import word_tokenize
import re
import string
import numpy as np


def get_corpus():
    sentence1 = "THIS is a random sentence. Barack Obama was the best president ever. OBAMA OBAMA OBAMA"
    sentence2 = "this is a serious sentence."
    sentence_list = []
    sentence_list.append(sentence1)
    sentence_list.append(sentence2)
    return sentence_list


def remove_punctuation(doc):
    # we remmove the punctuation signs: (, . : ;)
    remove_punct_regex = re.compile('[%s]' % re.escape(string.punctuation))

    tokenized_doc = word_tokenize(doc)
    tokenized_doc_no_punctuation = []
    for word in tokenized_doc:
        word = re.sub(remove_punct_regex, '', word)
        tokenized_doc_no_punctuation.append(word)

    return ' '.join(tokenized_doc_no_punctuation)


def remove_stop_words(doc):
    stop_words = stopwords.words('english')
    tokenized_doc = word_tokenize(doc)

    tokenized_doc_no_stop_words = [word for word in tokenized_doc if word not in stop_words]
    return ' '.join(tokenized_doc_no_stop_words)


def clean_corpus(corpus):
    cleaned_corpus = []

    for doc in corpus:
        doc = doc.lower()
        doc = remove_punctuation(doc)
        doc = remove_stop_words(doc)
        cleaned_corpus.append(doc)

    return cleaned_corpus


def add_words_to_frequencies(frequencies, doc):
    tokenized_doc = word_tokenize(doc)
    for word in tokenized_doc:
        if word in frequencies:
            frequencies[word] += 1
        else:
            frequencies[word] = 1

    return frequencies


def add_words_to_vector(words_vector, doc):
    tokenized_doc = word_tokenize(doc)
    for word in tokenized_doc:
        if not word in words_vector:
            words_vector.append(word)

    return words_vector


def get_terms_frequencies(doc):
    tokenized_doc = word_tokenize(doc)
    doc_size = tokenized_doc.__len__()
    frequencies = {}
    for word in tokenized_doc:
        if word in frequencies:
            frequencies[word] += 1
        else:
            frequencies[word] = 1

    for key, val in frequencies.items():
        frequencies[key] = val / doc_size

    return frequencies


def process_corpus():
    frequencies = {}
    words_vector = []

    corpus = get_corpus()
    corpus = clean_corpus(corpus)

    # we get full frequencies and the original word vector in the correct order
    for doc in corpus:
        frequencies = add_words_to_frequencies(frequencies, doc)
        words_vector = add_words_to_vector(words_vector, doc)

    print(words_vector)
    # print(frequencies)

    vectors = []
    for doc in corpus:
        current_vector = []
        doc_terms_frequencies = get_terms_frequencies(doc)
        print(doc_terms_frequencies)
        for word in words_vector:
            frequency = doc_terms_frequencies.get(word)
            frequency = 0 if frequency is None else frequency
            current_vector.append(frequency)
        vectors.append(current_vector)

    return vectors


if __name__ == "__main__":
    # query = input('Please insert your query:')
    # print(query)

    vectors = process_corpus()
    # TODO : CALCULATE IDF
    print(vectors)
