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

filename = "../resources/rowling_corpus.txt_processed"
obj = functions_autoencoder.Autoencoder(20)
raw_text = obj.setup_source(filename)
encoder, sequence_autoencoder = obj.build_autoencoder_model()
#filepath = "train_5/autoencoder-weights-improvement-99--0.0036.hdf5"
filepath = "train_rowling"
dataX, dataY = obj.prepare_dataset(raw_text)
#obj.resume_run(dataX, dataX,filepath)
obj.run(dataX,dataY,filepath)