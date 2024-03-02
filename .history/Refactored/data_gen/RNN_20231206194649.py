import tensorflow as tf
import chess
import numpy as np
from gen_utils import load_game_RNN
def get_model():
    input = tf.keras.layers.Input((None,8*8*18))
    dense1 = tf.keras.layers.Dense(20)(input)
    lstm = tf.keras.layers.LSTM(20, return_sequences=True)(dense1)
    model = tf.keras.Model(inputs = input, outputs = lstm)
    return model


def data_generator(batch_size):
    in_pgn = "human2.pgn"
    with open(in_pgn) as f:
        while True:
            batch = [load_game_RNN(chess.pgn.read_game(f)) for _ in range(batch_size)]
            min_len = min([len(elt) for elt in batch])
            print(min_len)
            x_same_len = np.array([[extract[0]  for extract in game[-min_len:]] for game in batch])
            y_same_len = np.array([[extract[0]  for extract in game[-min_len:]] for game in batch])
            x = [np.array(batch[i][-min_len:][0]).flatten() for i in range(batch_size)]
            y = [np.array(batch[i][-min_len:][1]).flatten()  for i in range(batch_size)]
            yield x, y
            


model = get_model()
model.compile(optimizer = "adam", loss= 'MSE')
data_gen = data_generator(32)
model.fit(data_gen)