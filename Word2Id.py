#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Shared by http://www.tensorflownews.com/
# Github:https://github.com/TensorFlowNews


from os import path
import numpy as np
import pickle

class Word2Id(object):
	def get_content(self,sentences):
		'''
		if (path.isfile("./word2id/content.pkl")):
			with open('./word2id/content.pkl', 'rb') as f:
				content = pickle.load(f)
				print(content)
		else:
			content = []
		'''
		content = []
		for sentence in sentences:
			content.append(sentence)
		with open('./word2id/content.pkl', 'wb') as f:
			pickle.dump(content, f)
		return content

	def build_vocabulary(self,content):
		if (path.isfile("./word2id/vocabulary.pkl")):
			with open('./word2id/vocabulary.pkl', 'rb') as f:
				vocabulary = pickle.load(f)
				vocabulary=dict(vocabulary)
				print(vocabulary)
		else:
			vocabulary = {}
		vocabulary = {}
		for sentence in content:
			for word in sentence:
				if word in vocabulary.keys():
					vocabulary[word]+=1
				else:
					vocabulary[word] = 1
		vocabulary=sorted(vocabulary.items(), key=lambda d:d[1],reverse=True)
		vocabulary = dict(vocabulary)
		with open('./word2id/vocabulary.pkl', 'wb') as f:
			pickle.dump(vocabulary, f,protocol=pickle.HIGHEST_PROTOCOL)
		return vocabulary

	def word2id(self,vocabulary):
		word2id = {}
		id=1
		for key in vocabulary:
			word2id[key]=id
			id+=1
		with open('./word2id/word2id.pkl', 'wb') as f:
			pickle.dump(word2id, f,protocol=pickle.HIGHEST_PROTOCOL)
		return word2id

	def id2word(self,word2id):
		id2word = dict((v, k) for k, v in word2id.items())
		with open('./word2id/id2word.pkl', 'wb') as f:
			pickle.dump(id2word, f,protocol=pickle.HIGHEST_PROTOCOL)
		return id2word

	def sentences2ids(self,sentences):
		with open('./word2id/word2id.pkl', 'rb') as f:
			word2id = pickle.load(f)
			word2id = dict(word2id)
		sentences_ids=[]
		for sentence in sentences:
			sentence_ids=[]
			for word in sentence:
				sentence_ids.append(word2id[word])
			sentences_ids.append(sentence_ids)
		return sentences_ids


