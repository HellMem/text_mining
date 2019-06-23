import matplotlib.pyplot as plt
import matplotlib
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk import word_tokenize
import re
import string
import numpy as np


def get_corpus():
    # sentence1 = "THIS is a random sentence. Barack Obama was the best president ever. OBAMA OBAMA OBAMA"
    # sentence2 = "this is a serious sentence."
    doc1 = open("Doc1.txt").read()
    doc2 = open("Doc2.txt").read()
    sentence_list = []
    sentence_list.append(doc1)
    sentence_list.append(doc2)
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
        # doc = remove_stop_words(doc)
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


def get_corpus_tf_and_idf(corpus):
    words = []

    corpus = clean_corpus(corpus)

    # we get full frequencies and the original word vector in the correct order
    for doc in corpus:
        words = add_words_to_vector(words, doc)

    tf_vectors = []
    idf_vector = [0] * words.__len__()

    for doc in corpus:
        current_vector = []
        doc_terms_frequencies = get_terms_frequencies(doc)
        for word in words:
            frequency = doc_terms_frequencies.get(word)
            frequency = 0 if frequency is None else frequency
            current_vector.append(frequency)
            if doc.__contains__(word):
                word_index = words.index(word)
                idf_vector[word_index] += 1

        tf_vectors.append(current_vector)

    idf_vector = np.array(idf_vector)
    idf_vector = idf_vector / corpus.__len__()
    idf_vector = np.log10(idf_vector)

    return [tf_vectors, idf_vector.tolist(), words]


def get_query_tf(query, words):
    query = query.lower()
    query = remove_punctuation(query)
    #query = remove_stop_words(query)

    words_query_tf = [0] * words.__len__()
    full_query_tf = get_terms_frequencies(query)

    for i in range(words.__len__()):
        word = words[i]
        frequency = full_query_tf.get(word)
        frequency = 0 if frequency is None else frequency
        words_query_tf[i] = frequency

    return words_query_tf


if __name__ == "__main__":
    doc_corpus = get_corpus()
    [tf, idf, words_vector] = get_corpus_tf_and_idf(doc_corpus)

    tf_idf = []
    for tf_doc in tf:
        tf_idf.append((np.array(tf_doc) * np.array(idf)).tolist())
    print('tf-idf:')
    print(tf_idf)
    print('tf:')
    print(tf)

    # query = 'is there a truck in the highway?'  # input('Please insert your query:')
    query = 'car driven on road'
    query_tf = get_query_tf(query, words_vector)
    #print(query_tf)
    # print(np.array(query_tf) * np.array(idf))
