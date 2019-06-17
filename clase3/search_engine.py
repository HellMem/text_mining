import matplotlib.pyplot as plt
import matplotlib
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk import word_tokenize
import re
import string


def get_corpus():
    sentence1 = "THIS is a random sentence. Barack Obama was the best president ever. OBAMA OBAMA OBAMA"
    # sentence2 = "this is a serious sentence."
    sentence_list = []
    sentence_list.append(sentence1)
    # sentence_list.append(sentence2)
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


def get_doc_tf_dictionary(doc):
    dictionary = {}

    tokenized_doc = word_tokenize(doc)
    for word in tokenized_doc:
        if word in dictionary:
            dictionary[word] += 1
        else:
            dictionary[word] = 1

    return dictionary


def process_corpus():
    vectors = []

    corpus = get_corpus()

    for doc in corpus:
        # cleaning
        doc = doc.lower()
        doc = remove_punctuation(doc)
        doc = remove_stop_words(doc)
        # cleaning
        print(doc)
        doc_tf_dictionary = get_doc_tf_dictionary(doc)

        print(doc_tf_dictionary)
        #TODO: Implement IDF -> This is mandatory on one model, not mandatory on the other

        #TODO: Implement Cosine Normalization

    return vectors


if __name__ == "__main__":
    # query = input('Please insert your query:')
    # print(query)

    process_corpus()
