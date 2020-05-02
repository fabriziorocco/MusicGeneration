import keras
import os
import glob
import sys
import datetime
import pickle
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, TensorBoard
from numpy import loadtxt
import numpy as np
import numpy
import matplotlib.pyplot as plt
from keras.models import load_model
from collections import Counter
#pip install keras-self-attention
from keras_self_attention import SeqSelfAttention

def Getting_notes():
    folder = os.path.join('/MusicGeneration/Dataset')

    notes = []

    os.chdir(folder)
    for file in glob.glob("*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_parse = None

        try:
            s2 = instrument.partitionByInstrument(midi)
            notes_parse = s2.parts[0].recurse()
        except:
            notes_parse = midi.flat.notes

        for element in notes_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    outfile = open('data_notes_updated.obj', 'wb')
    pickle.dump(notes, outfile)

    return notes


def Analyze_notes(notes_list):
    array_lenght = len(notes_list)

    freq = dict(Counter(notes_list))

    no = [count for _, count in freq.items()]
    plt.figure(figsize=(5, 5))
    plt.hist(no)

    all_notes = print('our Dataset contains ' + str(array_lenght) + ' notes and chords')

    return all_notes

def Prepare_network_sequences(notes_list, n_vocab):

    pitchnames = sorted(set(item for item in notes_list))

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    input_seq = []
    output_seq = []
    seq_length = 100

    for i in range(0, len(notes_list) - seq_length, 1):
        sequence_in = notes_list[i:i + seq_length]
        sequence_out = notes_list[i + seq_length]
        input_seq.append([note_to_int[char] for char in sequence_in])
        output_seq.append(note_to_int[sequence_out])

    n_patterns = len(input_seq)

    input_seq = np.reshape(input_seq, (n_patterns, seq_length, 1))
    input_seq = input_seq / float(n_vocab)

    output_seq = np_utils.to_categorical(output_seq)

    return (input_seq, output_seq)


def create_network(input_seq, n_vocab):
    model = Sequential()
    model.add(Bidirectional(LSTM(
        128,
        input_shape=(input_seq.shape[1], input_seq.shape[2]),
        return_sequences=True
    )))
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab, activation='relu'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


def train(model, input_seq, output_seq):
    tensorboard = TensorBoard(
        log_dir='.\logs',
        histogram_freq=0,
        write_images=False,
        write_graph=True,
        update_freq='epoch'
    )
    keras_callbacks = [tensorboard]

    model.fit(input_seq, output_seq, epochs=50, batch_size=128, callbacks=keras_callbacks)
    model.save("/MusicGeneration")
    model.summary()

def Start():
    #BE CAREFUL TO CHANGE PROPERLY THE FOLLOWING STRING.
    notes = Getting_notes() # IF FIRST RUN
    filehandler_def = open('/MusicGeneration/data_notes_updated.obj', 'rb')
    notes = pickle.load(filehandler_def)
    n_vocab = len(set(notes))

    input_notes_lenght = Analyze_notes(notes)

    network_input, network_output = Prepare_network_sequences(notes, n_vocab)

    print(network_input, network_output)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)

if __name__ == '__main__':
    Start()