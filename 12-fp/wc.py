#-*- coding: utf-8 -*-
from os import path
import matplotlib.pyplot as plt
from wordcloud import WordCloud,ImageColorGenerator
from scipy.misc import imread
font_path = 'simhei.ttf'
back_coloring_path = "z1.jpg" # 设置背景图片路径
wordDic = {}
wordlist = open('result.txt').readlines()
for line in wordlist:
    line = line.split(',')
    #print(line[0])
    wordDic[line[0]]=float(line[1])
back_coloring = imread(back_coloring_path)    
wc = WordCloud(mask=back_coloring,font_path=font_path,background_color="white", max_words=521, max_font_size=100, random_state=42,width=1000, height=860, margin=2)
wc.generate_from_frequencies(wordDic)
image_colors = ImageColorGenerator(back_coloring)

plt.figure()
plt.imshow(wc)
plt.axis("off")
plt.show()
