#functions_autoencoder.py
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
DUMMY = ["END_TOKEN"]
class Autoencoder:

    def __init__(self,seq_length):
        self.seq_length = seq_length
        self.embedding_vector_size = 100
        self.n_words = 0
        self.n_vocab = 0

        self.embeddings_index = models.Word2Vec.load("../resources/data_corpus_gensim_model")
        print('Found %s word vectors.' % len(self.embeddings_index.vocab))
        #print embeddings_index.vocab
        #print embeddings_index.most_similar("know")[0][0]

    def display(self):
        pass
        

    def setup_source(self,filename):
        # load utf-8 text and covert to lowercase
        #filename = "../resources/sentence_corpus_rowling"
        raw_text = open(filename).read()
        raw_text = nltk.sent_tokenize(raw_text.lower())
        self.n_sentences = len(raw_text)
        print "Total Sentences for Training: ", self.n_sentences
        # print raw_text[0][:10]
        return raw_text

    # prepare the dataset of input to output pairs encoded as integers
    def prepare_dataset(self,raw_text):
        dataX = []
        dataY = []
        for sent in raw_text[:5000]:
            seq_in = self.frame_input(sent)
            tagged_sent = dict(pos_tag(seq_in))
            record = []
            for word in seq_in:
                try:
                    if tagged_sent[word]!="NNP":
                        record.append(self.embeddings_index[word])
                    else:
                        raise KeyError
                except:
                    record.append(self.embeddings_index["harry"])        #consider "harry" as replacement of all proper noun
            dataX.append(record)
        return dataX,dataY

    def build_autoencoder_model(self):
        
        inputs = Input(shape=(self.seq_length, self.embedding_vector_size))
        encoded = LSTM(300)(inputs)

        decoded = RepeatVector(self.seq_length)(encoded)
        decoded = LSTM(100, return_sequences=True)(decoded)

        #sequence_autoencoder = keras.models.load_model("train_3/autoencoder-weights-improvement-49--0.0057.hdf5")
        sequence_autoencoder = Model(inputs, decoded)
        encoder = Model(inputs, encoded)

        encoder.compile(optimizer='adadelta', loss='cosine_proximity')

        sequence_autoencoder.compile(optimizer='adadelta', loss='cosine_proximity')
        self.mymodel = sequence_autoencoder
        self.mymodel2 = encoder
        return encoder,sequence_autoencoder
        
    def run(self, dataX, dataY, filepath):
        
        n_patterns = len(dataX)
        print "Total Patterns: ", n_patterns
        filepath = filepath+"/autoencoder-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"

        X = numpy.reshape(dataX, (n_patterns, self.seq_length, self.embedding_vector_size))

        # define the checkpoint
        #filepath="train_4/autoencoder-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        self.mymodel2.save("train_5/model2.hdf5")

        # and trained it via:
        self.mymodel.fit(X,X,nb_epoch=100, batch_size=10, shuffle=True, callbacks=callbacks_list)

    def resume_run(self, dataX, dataY, filepath):
        self.mymodel = keras.models.load_model(filepath)
        self.run(dataX, dataY, filepath.split('/')[0])

    def setup_test(self,model_filename):
        self.mymodel = keras.models.load_model(model_filename)
        self.mymodel2 = keras.models.load_model("train_5/model2.hdf5")
        self.mymodel.load_weights(model_filename)

    def test(self,input_sentence):
        input_sentence = self.frame_input(input_sentence)
        pattern = []
        print "Input Sentence : "," ".join(input_sentence)
        for w in input_sentence:
            try:
                pattern.append(self.embeddings_index[w])
            except:
                pattern.append(self.embeddings_index["harry"])
        x = numpy.reshape(pattern, (1, len(pattern), self.embedding_vector_size))
        prediction = self.mymodel.predict(x,batch_size=1)
        prediction2 = self.mymodel2.predict(x,batch_size=1)
        print prediction2.shape
        print "Output Sentence : \"", ' '.join([self.embeddings_index.most_similar(positive=[value], topn=1)[0][0] for value in prediction[0]]), "\""
        return "\"", ' '.join([self.embeddings_index.most_similar(positive=[value], topn=1)[0][0] for value in prediction[0]]), "\""


    def frame_input(self,sentence):
        s = nltk.word_tokenize(sentence)
        if len(s) < self.seq_length : #if sent length smaller than seq_length, add dummy tokens
            s.extend(DUMMY*(self.seq_length-len(s)))
        else :  #if sentence length ore than model seq_length, chop it to seq_length
            s = s[:self.seq_length]
        return s
