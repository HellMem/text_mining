import nltk
import numpy
import scipy
import matplotlib
from nltk.tokenize import sent_tokenize

nltk.download()

'''

text = "Welcome readers. I hope you find it interesting. Please do reply."
print(text)

print(sent_tokenize(text))

with open("Example1.txt") as doc1: #por default se abre como sólo lectura doc = open("Example1.txt", 'r')
    text = doc1.read()
    print(sent_tokenize(text))
doc1.close()

'''