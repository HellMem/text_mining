from wordcloud import WordCloud
import matplotlib.pyplot as plt

doc = open("Example1.txt")
text = doc.read()
wordcloud = WordCloud(max_font_size=40).generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()