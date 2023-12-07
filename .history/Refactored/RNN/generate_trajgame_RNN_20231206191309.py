from Refactored.data_gen.gen_TC import generate_batch
import tensorflow as tf
import chess
from Refactored.data_gen.gen_utils import load_game_RNN



def get_model():
    input = tf.keras.layers.Input(())


def data_generator(batch_size):
    in_pgn = "human2.pgn"
    with open(in_pgn) as f:
        while True:
            batch = [load_game_RNN(chess.pgn.read_game(f)) for _ in range(batch_size)]
            min_len = min([len(elt) for elt in batch])
            yield [batch[i][:min_len][0] for i in range(batch_size)], [batch[i][:min_len][1] for i in range(batch_size)]
            
            
