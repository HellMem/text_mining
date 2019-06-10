import matplotlib.pyplot as plt
import matplotlib
from wordcloud import WordCloud
from nltk import word_tokenize
import re
import string

if __name__ == "__main__":
    matplotlib.use('TkAgg')


    doc = open("Example1.txt")

    tokenized_doc = [word_tokenize(text) for text in doc]

    #we remmove the punctuation signs: (, . : ;)
    x = re.compile('[%s]' % re.escape(string.punctuation))

    tokenized_doc_no_punctuation = []
    for sent in tokenized_doc:
        new_sent = []
        for token in sent:
            new_token = re.sub(x, '', token)
            if not new_token == '':
                new_sent.append(new_token)
        tokenized_doc_no_punctuation.append(new_sent)
    print(tokenized_doc_no_punctuation)

    doc = tokenized_doc_no_punctuation

    new_text_array = []
    for sent in doc:
        for word in sent:
            new_text_array.append(word)

    new_text = ' '.join(new_text_array)

    print(new_text)

    wordcloud = WordCloud(max_font_size=40).generate(new_text)

    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
