from Refactored.data_gen.gen_TC import generate_batch
import tensorflow as tf
import chess
from Refactored.data_gen.gen_utils import load_game



def get_model():
    input = tf.keras.layers.Input(())


def data_generator(batch_size):
    in_pgn = "human2.pgn"
    with open(in_pgn) as f:
        while True:
            batch = [load_game(chess.pgn.read_game(f)) for _ in range(batch_size)]
            yield [batch[i][0] for i in range(batch_size)], [batch[i][0] for i in range(batch_size)]
            
            
