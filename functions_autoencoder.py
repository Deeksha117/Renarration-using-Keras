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
        self.sentence_embedding_length = 1000

        self.embeddings_index = models.Word2Vec.load("../resources/data_corpus_gensim_model_with_sent_tokens")
        print('Found %s word vectors.' % len(self.embeddings_index.vocab))
        # exit()
        # print self.embeddings_index["SENT_PAD"], self.embeddings_index["SENT_START"], self.embeddings_index["SENT_END"]
        #print embeddings_index.vocab
        #print embeddings_index.most_similar("know")[0][0]

    def display(self,text,prediction):
        print text
        print  ' '.join([self.embeddings_index.most_similar(positive=[value], topn=1)[0][0] for value in prediction[0]]), "\""
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
        for sent in raw_text:
            seq_in = self.frame_input(sent)
            tagged_sent = dict(pos_tag(seq_in))
            record = []
            for word in seq_in:
                try :
                    record.append(self.embeddings_index[word])
                except:
                    print "invalid word",word
                    record.append(self.embeddings_index["harry"])        #consider "harry" as replacement of all proper noun
            
            dataX.append(record)
        return dataX,dataY

    def build_autoencoder_model(self):
        
        inputs = Input(shape=(self.seq_length, self.embedding_vector_size))
        encoded = LSTM(self.sentence_embedding_length)(inputs)

        decoded1 = RepeatVector(self.seq_length)(encoded)
        decoded = LSTM(self.embedding_vector_size, return_sequences=True)(decoded1)

        sequence_autoencoder = Model(inputs, decoded)
        encoder = Model(inputs, encoded)

        encoded_input = Input(shape=(self.seq_length,self.sentence_embedding_length,))
        decoder_layer = sequence_autoencoder.layers[-1]
        decoder = Model(encoded_input, decoder_layer(encoded_input))

        # inter_input = Input(shape=(300,))
        # inter_layer = sequence_autoencoder.layers[1]
        intermediate = Model(inputs, decoded1)

        encoder.compile(optimizer='adadelta', loss='cosine_proximity')
        decoder.compile(optimizer='adadelta', loss='cosine_proximity')
        intermediate.compile(optimizer='adadelta', loss='cosine_proximity')
        sequence_autoencoder.compile(optimizer='adadelta', loss='cosine_proximity')

        self.mymodel = sequence_autoencoder
        self.encoder = encoder
        self.decoder = decoder
        self.inter = intermediate
        return encoder, sequence_autoencoder
        
    def run(self, dataX, dataY, filepath):
        
        n_patterns = len(dataX)
        print "Total Patterns: ", n_patterns
        filepath1 = filepath+"/autoencoder-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
        filepath2 = filepath+"/encodermodel.hdf5"
        filepath3 = filepath+"/decodermodel.hdf5"
        filepath4 = filepath+"/intermodel.hdf5"

        X = numpy.reshape(dataX, (n_patterns, self.seq_length, self.embedding_vector_size))

        # define the checkpoint
        checkpoint = ModelCheckpoint(filepath1, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]

        # and trained it via:
        self.mymodel.fit(X,X,nb_epoch=100, batch_size=10, shuffle=True, callbacks=callbacks_list)
        self.encoder.save(filepath2)
        self.decoder.save(filepath3)
        self.inter.save(filepath4)

    def resume_run(self, dataX, dataY, filepath):
        filepath1 = filepath
        filepath = filepath.split("/")[0]
        # filepath2 = filepath+"/encodermodel.hdf5"
        # filepath3 = filepath+"/decodermodel.hdf5"
        # filepath4 = filepath+"/intermodel.hdf5"

        self.mymodel = keras.models.load_model(filepath1)
        # self.encoder = keras.models.load_model(filepath2)
        # self.decoder = keras.models.load_model(filepath3)
        # self.inter = keras.models.load_model(filepath4)
        self.run(dataX, dataY, filepath)

    def setup_test(self,model_filename):
        self.mymodel = keras.models.load_model(model_filename)
        self.encoder = keras.models.load_model("train_5/encodermodel.hdf5")
        self.decoder = keras.models.load_model("train_5/decodermodel.hdf5")
        self.inter = keras.models.load_model("train_5/intermodel.hdf5")
        # self.mymodel.load_weights(model_filename)

    def test(self,input_sentence):
        input_sentence = self.frame_input(input_sentence)
        pattern = []
        print "Input Sentence : "," ".join(input_sentence)
        for w in input_sentence:
            pattern.append(self.embeddings_index[w])

        x = numpy.reshape(pattern, (1, len(pattern), self.embedding_vector_size))
        prediction = self.mymodel.predict(x,batch_size=1)
        self.display("sequence autoencoder",prediction)

        # prediction1 = self.encoder.predict(x,batch_size=1)
        # prediction2 = numpy.tile(prediction1,(self.seq_length,1))
        # prediction2 = prediction2[None,...]

        prediction3 = self.inter.predict(x,batch_size=1)

        # print prediction2, prediction3

        # prediction4 = self.decoder.predict(prediction2,batch_size=1)
        prediction5 = self.decoder.predict(prediction3,batch_size=1)
        # self.display("decoder output",prediction4)
        self.display("decoder 2 output",prediction5)
        return "\"", ' '.join([self.embeddings_index.most_similar(positive=[value], topn=1)[0][0] for value in prediction[0]]), "\""

    def setup_test_encoder_decoder(self, encoderfile, decoderfile):
        self.encoder = keras.models.load_model(encoderfile)
        self.decoder = keras.models.load_model(decoderfile)

    def test_encoder_decoder(self, input_sentence):
        input_sentence = self.frame_input(input_sentence)
        pattern = []
        print "Input Sentence : "," ".join(input_sentence)
        for w in input_sentence:
            try:
                pattern.append(self.embeddings_index[w])
            except:
                pattern.append(self.embeddings_index["harry"])

        x = numpy.reshape(pattern, (1, len(pattern), self.embedding_vector_size))
        prediction = self.decoder.predict(self.encoder.predict(x,batch_size=1),batch_size=1)
        self.display("sequence autoencoder",prediction)


    def frame_input(self,sentence):
        s = nltk.word_tokenize(sentence)
        if len(s) > self.seq_length-2:
            s = s[:self.seq_length-2]
        s = ["SENT_START"] + s + ["SENT_END"]
        s.extend(["SENT_PAD"]*(self.seq_length-len(s)))
        return s
