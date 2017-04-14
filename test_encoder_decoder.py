from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model
import numpy
import nltk
from nltk.tag import pos_tag
import sys
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.layers import Input, Embedding, LSTM, Dense, merge
from keras.models import Model
from gensim import models
import os
import functions_autoencoder

reload(sys)  
sys.setdefaultencoding('utf8')

filename = "../resources/austen_corpus.txt"
obj = functions_autoencoder.Autoencoder(20)

raw_text = obj.setup_source(filename)

encoderfile = "train_austen/intermodel.hdf5"
decoderfile = "train_rowling/decodermodel.hdf5"
obj.setup_test_encoder_decoder(encoderfile,decoderfile)

# generate words
for i in range(10):
	# pick a random seed
	start = numpy.random.randint(0, len(raw_text))
	pattern = raw_text[start]
	#print embeddings_index.most_similar(positive=[pattern[0]], topn=1)
	obj.test_encoder_decoder(pattern)
	#obj.test("dudley 's hands jerked upward to tower his mouth.")
	print "Test ", i
print "\nDone."