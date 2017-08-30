#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Shared by http://www.tensorflownews.com/
# Github:https://github.com/TensorFlowNews


import nltk
from nltk.tokenize import RegexpTokenizer
from Word2Id import Word2Id

text='''
绝代有佳人，幽居在空谷。
自云良家子，零落依草木。
关中昔丧乱，兄弟遭杀戮。
官高何足论，不得收骨肉。
世情恶衰歇，万事随转烛。
夫婿轻薄儿，新人美如玉。
合昏尚知时，鸳鸯不独宿。
但见新人笑，那闻旧人哭！
在山泉水清，出山泉水浊。
侍婢卖珠回，牵萝补茅屋。
摘花不插发，采柏动盈掬。
天寒翠袖薄，日暮倚修竹。
'''

text_tokenize=text.split("。")
print(text_tokenize)
content=[]
for sent in text_tokenize:
	sentence=[]
	for word in sent:
		#print(word)
		sentence.append(word)
	content.append(sentence)


w2i=Word2Id()

content=w2i.get_content(content)

print("content",content)


vocabulary=w2i.build_vocabulary(content)
print(len(vocabulary))
word2id=w2i.word2id(vocabulary)
print(word2id)
id2word=w2i.id2word(word2id)
print(id2word)
sentences_ids=w2i.sentences2ids(content)
print(sentences_ids)
for sentence in sentences_ids:
	if(len(sentence)>20):
		sentence=sentence[0:20]
	for i in range(0,20-len(sentence)):
		sentence.append(0)
	sentence=sentence.reverse()

print(sentences_ids)
print(len(sentences_ids))