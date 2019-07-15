from nltk.corpus import stopwords
from nltk import word_tokenize
import re
import string
import numpy as np
import os


def get_corpus(path):
    sentence_list = []
    for r, d, f in os.walk(path):
        for file in f:
            sentence_list.append(open(path + file).read())
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
    idf_vector = np.log10(idf_vector) * -1

    return [tf_vectors, idf_vector.tolist(), words]


def get_query_tf_tfidf(query, words, tf, tf_idf):
    words_query_tfidf = [0] * words.__len__()
    words_query_tf = [0] * words.__len__()

    for i in range(words.__len__()):
        word = words[i]
        if query.__contains__(word):
            for current_tf_idf, current_tf in zip(tf_idf, tf):
                if current_tf_idf[i] > words_query_tfidf[i]:
                    words_query_tfidf[i] = current_tf_idf[i]
                if current_tf[i] > words_query_tf[i]:
                    words_query_tf[i] = current_tf[i]

    return [words_query_tfidf, words_query_tf]


def get_euclidean_similarity(docs_idf, query_idf):
    vq = np.array(query_idf)
    euclidean_similarities = []

    for i in range(docs_idf.__len__()):
        vi = np.array(docs_idf[i])
        d_q_vi = np.linalg.norm(vi - vq)
        euclidean_similarities.append(d_q_vi)

    return euclidean_similarities


def get_cosine_similarity(docs_idf, query_idf):
    v_q = np.array(query_idf)
    n_v_q = np.linalg.norm(v_q)
    cosine_similarities = []

    for i in range(docs_idf.__len__()):
        v_i = np.array(docs_idf[i])
        n_v_i = np.linalg.norm(v_i)

        dot_q_vi = np.dot(v_q, v_i)

        d_q_vi = dot_q_vi / (n_v_q * n_v_i)
        cosine_similarities.append(d_q_vi)

    return cosine_similarities


def process_query(query, words_vector, tf, tf_idf):
    print('\n\nquery: ')
    print(query)
    [words_query_tfidf, words_query_tf] = get_query_tf_tfidf(query, words_vector, tf, tf_idf)
    print('tf-idf')
    print(words_query_tfidf)
    print('tf')
    print(words_query_tf)

    print('Similarity by Euclidean (min): ')
    euclidean_similarities_query = get_euclidean_similarity(tf_idf, words_query_tfidf)
    print(euclidean_similarities_query)
    print('Top 1 most relevant with Euclidean Doc', np.argmin(euclidean_similarities_query) + 1)

    print('Similarity by Cosine (max): ')
    cosine_similarities_query = get_cosine_similarity(tf_idf, words_query_tfidf)
    print(cosine_similarities_query)
    print('Top 1 most relevant with Cosine Doc', np.argmax(cosine_similarities_query) + 1)


if __name__ == "__main__":
    # path = "files/" example path
    path = input("Input corpus directory: ")
    doc_corpus = get_corpus(path)
    [tf, idf, words_vector] = get_corpus_tf_and_idf(doc_corpus)

    tf_idf = []
    for tf_doc in tf:
        tf_idf.append((np.array(tf_doc) * np.array(idf)).tolist())

    print('tf-idf:')
    print(tf_idf)
    print('tf:')
    print(tf)

    # query1 = 'is there a truck in the highway?'
    # process_query(query1, words_vector, tf, tf_idf)

    # query2 = 'car driven on road'
    # process_query(query2, words_vector, tf, tf_idf)

    while True:
        query = input("\n\nInput your search: ")
        process_query(query, words_vector, tf, tf_idf)
