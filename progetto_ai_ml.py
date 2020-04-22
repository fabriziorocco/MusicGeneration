
#########################################################################

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


model = Sequential()
model.add(Bidirectional(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]), 
        recurrent_dropout=0.3,
        return_sequences=True
    )))
model.add(LSTM(512))
model.add(BatchNorm())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNorm())
model.add(Dropout(0.3))
model.add(Dense(n_vocab))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Model Checkpoint
filepath = os.path.abspath("weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5")
keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='min')
callbacks_list = [checkpoint]

model.fit(network_input, network_output, epochs=50, batch_size=128, callbacks=callbacks_list)

print(model.summary())