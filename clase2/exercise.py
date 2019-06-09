import matplotlib.pyplot as plt
# from wordcloud import WordCloud
from nltk.probability import FreqDist
import matplotlib
from nltk.tokenize import sent_tokenize
import nltk

matplotlib.use('TkAgg')

fd = FreqDist()
doc = open("Example1.txt")
#text_words = doc.read().split()
#print(text_words)
sentences = sent_tokenize(doc.read())
words = nltk.word_tokenize(sentences)

print(words)


'''
for word in text_words:
    fd[word] += 1

ranks = []
freqs = []
for rank, word in enumerate(fd):
    ranks.append(rank + 1)
    freqs.append(fd[word])

fd_dict = {}
for rank, word in enumerate(fd):
    fd_dict[word] = fd[word]


wordcloud = WordCloud().generate_from_frequencies(fd_dict, 40)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
'''
