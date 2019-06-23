import matplotlib.pyplot as plt
import matplotlib
from wordcloud import WordCloud
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
import string

if __name__ == "__main__":
    matplotlib.use('TkAgg')
    stop_words = stopwords.words('english')

    doc = open("Example1.txt")

    tokenized_doc = [word_tokenize(text) for text in doc]

    # we remmove the punctuation signs: (, . : ;)}
    # we remove stop words
    # we convert all to lower case
    x = re.compile('[%s]' % re.escape(string.punctuation))

    tokenized_doc_no_punctuation = []
    for sent in tokenized_doc:
        new_sent = []
        for token in sent:
            new_token = re.sub(x, '', token)
            if not new_token == '':
                if new_token not in stop_words:
                    new_sent.append(new_token.lower())
        tokenized_doc_no_punctuation.append(new_sent)

    doc = tokenized_doc_no_punctuation

    words_frequencies = {}
    for sent in doc:
        for word in sent:
            if word in words_frequencies:
                words_frequencies[word] += 1
            else:
                words_frequencies[word] = 1

    wordcloud = WordCloud(max_font_size=40).generate_from_frequencies(frequencies=words_frequencies)

    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
