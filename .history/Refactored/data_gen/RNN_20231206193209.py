import tensorflow as tf
import chess
from gen_utils import load_game_RNN
def get_model():
    input = tf.keras.layers.Input((None,None,8,8,18))
    flat = tf.keras.layers.Reshape((-1,))(input)
    dense1 = tf.keras.layers.Dense(20)(flat)
    lstm = tf.keras.layers.LSTM(20, return_sequences=True)(dense1)
    model = tf.keras.Model(inputs = input, outputs = lstm)
    return model


def data_generator(batch_size):
    in_pgn = "human2.pgn"
    with open(in_pgn) as f:
        while True:
            batch = [load_game_RNN(chess.pgn.read_game(f)) for _ in range(batch_size)]
            min_len = min([len(elt) for elt in batch])
            yield [batch[i][:min_len][0] for i in range(batch_size)], [batch[i][:min_len][1] for i in range(batch_size)]
            


model = get_model()
model.compile(optimizer = "adam", loss= 'MSE')
data_gen = data_generator(32)
model.fit(data_gen)