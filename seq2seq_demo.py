#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

import numpy as np
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

print(sentences_ids)

train_x=sentences_ids[0:-1]
train_y=sentences_ids[1:]
print(train_x)
print(train_y)

batch_size=2
sequence_length=20
num_encoder_symbols=128
num_decoder_symbols=128
embedding_size=16
learning_rate=0.001

encoder_inputs=tf.placeholder(dtype=tf.int32,shape=[batch_size,sequence_length])
decoder_inputs=tf.placeholder(dtype=tf.int32,shape=[batch_size,sequence_length])

logits=tf.placeholder(dtype=tf.float32,shape=[batch_size,sequence_length,num_decoder_symbols])
targets=tf.placeholder(dtype=tf.int32,shape=[batch_size,sequence_length])
weights=tf.placeholder(dtype=tf.float32,shape=[batch_size,sequence_length])


train_weights=np.ones(shape=[batch_size,sequence_length],dtype=np.float32)

cell=tf.nn.rnn_cell.BasicLSTMCell(sequence_length)

def seq2seq(encoder_inputs,decoder_inputs,cell,num_encoder_symbols,num_decoder_symbols,embedding_size):
	encoder_inputs = tf.unstack(encoder_inputs, axis=0)
	decoder_inputs = tf.unstack(decoder_inputs, axis=0)
	results,states=tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
    encoder_inputs,
    decoder_inputs,
    cell,
    num_encoder_symbols,
    num_decoder_symbols,
    embedding_size,
    output_projection=None,
    feed_previous=False,
    dtype=None,
    scope=None
)
	return results

def get_loss(logits,targets,weights):
	loss=tf.contrib.seq2seq.sequence_loss(
		logits,
		targets=targets,
		weights=weights
	)
	return loss

results=seq2seq(encoder_inputs,decoder_inputs,cell,num_encoder_symbols,num_decoder_symbols,embedding_size)
logits=tf.stack(results,axis=0)
print(logits)
loss=get_loss(logits,targets,weights)
train_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	count=0
	while(count<1000):
		count=count+1
		print("cout:",count)
		for step in range(0,2):
			print("step:",step)
			train_encoder_inputs=train_x[step*batch_size:step*batch_size+batch_size][:]
			train_decoder_inputs=train_y[step*batch_size:step*batch_size+batch_size][:]
			#results_value=sess.run(results,feed_dict={encoder_inputs:train_encoder_inputs,decoder_inputs:train_decoder_inputs})
			cost = sess.run(loss, feed_dict={encoder_inputs: train_encoder_inputs, targets: train_decoder_inputs,
			                                 weights:train_weights,decoder_inputs:train_decoder_inputs})
			print(cost)
			op = sess.run(train_op, feed_dict={encoder_inputs: train_encoder_inputs, targets: train_decoder_inputs,
			                                 weights: train_weights, decoder_inputs: train_decoder_inputs})
			step=step+1