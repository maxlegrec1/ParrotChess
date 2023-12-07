import tensorflow as tf
import chess
import numpy as np
from gen_utils import load_game_RNN
def get_model():
    input = tf.keras.layers.Input((None,8*8*18))
    dense1 = tf.keras.layers.Dense(20)(input)
    lstm = tf.keras.layers.LSTM(20, return_sequences=True)(dense1)
    output = tf.keras.layers.Dense(1860)(lstm)
    model = tf.keras.Model(inputs = input, outputs = output)
    return model


def data_generator(batch_size):
    in_pgn = "human2.pgn"
    nb_game = 0
    with open(in_pgn) as f:
        while True:
            batch = [load_game_RNN(chess.pgn.read_game(f)) for _ in range(batch_size)]
            nb_game+=batch_size
            min_len = min([len(elt) for elt in batch])
            x_same_len = np.array([[extract[0].flatten()  for extract in game[-min_len:]] for game in batch])
            y_same_len = np.array([[extract[1].flatten()  for extract in game[-min_len:]] for game in batch])
            print(nb_game)
            yield x_same_len, y_same_len
            


model = get_model()
model.compile(optimizer = "adam", loss= 'MSE')
data_gen = data_generator(16)
model.fit(data_gen)