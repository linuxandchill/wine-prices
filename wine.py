import os 
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.layers import Input, Dense
from keras.models import Model
import keras

#layers = keras.layers
data = pd.read_csv('wine_data.csv')

train_size = int(len(data) * .8)

#train features
description_train = data['description'][:train_size]
variety_train = data['variety'][:train_size]

#training labels
labels_train = data['price'][:train_size]

# Test features
description_test = data['description'][train_size:]
variety_test = data['variety'][train_size:]

# Test labels
labels_test = data['price'][train_size:]

###### WINE DESCRIPTION ######
vocab_size = 12000
tokenize = keras.preprocessing.text.Tokenizer(num_words=vocab_size, char_level=False)
tokenize.fit_on_texts(description_train)
descriptionBOWtrain = tokenize.texts_to_matrix(description_train)
descriptionBOWtest = tokenize.texts_to_matrix(description_test)


############ WINE VARIETY #########
encoder = LabelEncoder()
encoder.fit(variety_train)
variety_train = encoder.transform(variety_train)
variety_test = encoder.transform(variety_test)
num_classes = np.max(variety_train) + 1

#one hot labels
variety_train = keras.utils.to_categorical(variety_train, num_classes)
variety_test = keras.utils.to_categorical(variety_test, num_classes)

### BUILD WIDE MODEL ########
BOWinputs = layers.Input(shape=(vocab_size,))
variety_inputs = layers.Input(shape=(num_classes,))
merged_layer = layers.concatenate([BOWinputs, variety_inputs])
merged_layer = layers.Dense(256, activation='relu')(merged_layer)
predictions = layers.Dense(1)(merged_layer)

wide_model = Model(inputs=[BOWinputs, variety_inputs], outputs=predictions)

wide_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

######### EMBEDDING & DEEP MODEL ######
train_embed = tokenize.texts_to_sequences(description_train)
test_embed = tokenize.texts_to_sequences(description_test)

max_seq_len = 170
train_embed = keras.preprocessing.sequence.pad_sequences(train_embed, maxlen=max_seq_len)
test_embed = keras.preprocessing.sequence.pad_sequences(test_embed, maxlen=max_seq_len)

deep_inputs = layers.Input(shape=(max_seq_len, ))
embedding = layers.Embedding(vocab_size, 8, input_length=max_seq_len)(deep_inputs)
embedding = layers.Flatten()(embedding)
embed_out = layers.Dense(1, activation='linear')(embedding)
deep_model = Model(inputs=deep_inputs, outputs=embed_out)
deep_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

########### COMBINING BOTH MODELS #############
merged_out = layers.concatenate([wide_model.output, deep_model.output])
merged_out = layers.Dense(1)(merged_out)
combined_model = Model(wide_model.input + [deep_model.input], merged_out)
combined_model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])


### TRAIN 
combined_model.fit([descriptionBOWtrain, variety_train] + [train_embed], 
        labels_train, epochs=10, batch_size=128)
