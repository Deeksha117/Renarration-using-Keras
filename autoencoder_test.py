from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model
import numpy
import nltk
from nltk.tag import pos_tag
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.layers import Input, Embedding, LSTM, Dense, merge
from keras.models import Model
from gensim import models
reload(sys)  
sys.setdefaultencoding('utf8')

# load utf-8 text and covert to lowercase
filename = "../resources/austen_corpus"
raw_text = open(filename).read()
raw_text = nltk.word_tokenize(raw_text.lower())

embeddings_index = models.Word2Vec.load("../resources/data_corpus_gensim_model")
words=[]
for w in raw_text:
	#print w,embeddings_index.get(w)
	if w in embeddings_index.vocab:
		words.append(w)
words = set(words)

# summarize the loaded data
n_words = len(raw_text)
n_vocab = len(words)
print "Total Words: ", n_words
print "Total Vocab: ", n_vocab
#print embeddings_index["hermione"]
#arr = numpy.array(embeddings_index["know"],dtype=float)
#print embeddings_index.most_similar(positive=[embeddings_index["know"]], topn=1)

print('Found %s word vectors.' % len(embeddings_index.vocab))

# prepare the dataset of input to output pairs encoded as integers
seq_length = 10
dataX = []
dataY = []
for i in range(0, (n_words - seq_length)/2, 100):
	seq_in = raw_text[i:i + seq_length]
	record = []
	tagged_sent = dict(pos_tag(seq_in))
	for word in seq_in:
		try:
			if tagged_sent[word]!="NNP":
				record.append(embeddings_index[word])
			else:
				raise KeyError
		except:
			record.append(embeddings_index["harry"])		#consider "harry" as replacement of all proper noun
		dataX.append(record)
n_patterns = len(dataX)
print "Total Patterns: ", n_patterns
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 100))

inputs = Input(shape=(seq_length, 100))
encoded = LSTM(100)(inputs)

decoded = RepeatVector(seq_length)(encoded)
decoded = LSTM(100, return_sequences=True)(decoded)

sequence_autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)

# load the network weights
#filename = "weights-improvement-19-1.9435.hdf5"
filename="train_3/autoencoder-weights-improvement-33--0.0064.hdf5"
sequence_autoencoder.load_weights(filename)
#sequence_autoencoder.compile(loss='cosine_proximity', optimizer='adadelta')

example_sentence = "The event had every promise of happiness for her friend"
pattern = []
print example_sentence
for w in example_sentence.lower().split():
	pattern.append(embeddings_index[w])

x = numpy.reshape(pattern, (1, len(pattern), 100))
#print x
prediction = sequence_autoencoder.predict(x,batch_size=1)
print "\"", ' '.join([embeddings_index.most_similar(positive=[value], topn=1)[0][0] for value in prediction[0]]), "\""
print
exit()

# generate words
for i in range(100):

	# pick a random seed
	start = numpy.random.randint(0, len(dataX)-1)
	pattern = dataX[start]
	#print embeddings_index.most_similar(positive=[pattern[0]], topn=1)
	print "Seed:"
	#print "\"", ' '.join([str(value) for value in pattern]), "\""
	print "\"", ' '.join([embeddings_index.most_similar(positive=[value], topn=1)[0][0]  for value in pattern]), "\""
	
	x = numpy.reshape(pattern, (1, len(pattern), 100))
	#print x
	prediction = sequence_autoencoder.predict(x,batch_size=1)
	prediction1 = encoder.predict(x, batch_size=1)
	#print "prediction1: ", prediction1

	print "\"", ' '.join([embeddings_index.most_similar(positive=[value], topn=1)[0][0] for value in prediction[0]]), "\""
	print
print "\nDone."