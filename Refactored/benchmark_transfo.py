import tensorflow as tf
import numpy as np
from tqdm import tqdm
from typing import Tuple, List
from datetime import datetime

def residual(x,num_filter):
    skip= x
    x = tf.keras.layers.Conv2D(num_filter, (3, 3), padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(num_filter, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.add([skip, x])
    x = tf.keras.layers.ReLU()(x)
    return x

from create_transformer import Encoder
h = 12  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_ff = 2048  # Dimensionality of the inner fully connected layer
d_model = 256  # Dimensionality of the model sub-layers' outputs
n = 10  # Number of layers in the encoder stack
batch_size = 32  # Batch size from the training process
dropout_rate = 0.1  # Frequency of dropping the input units in the dropout layers
enc_vocab_size = 12*64 # Vocabulary size for the encoder
input_seq_length = 32  # Maximum length of the input sequence
transformer = Encoder(enc_vocab_size, input_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)

def create_A0(num_residuals):
    vocab = tf.keras.layers.Input(shape=(32))
    mask = tf.keras.layers.Input(shape=(32))
    y = transformer(vocab, mask, True)
    y = tf.keras.layers.Flatten()(y)
    input = tf.keras.Input(shape=(8, 8, 14))
    x = input
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(x)
    for _ in range(num_residuals):
        x = residual(x, 256)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same',activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.concatenate([x,y])
    x = tf.keras.layers.Dense(8192,activation='relu')(x)
    x = tf.keras.layers.Dense(8192,activation='relu')(x)
    x = tf.keras.layers.Dense(4096,activation='relu')(x)
    x = tf.keras.layers.Softmax()(x)
    output = x
    return tf.keras.Model(inputs=[input,vocab,mask], outputs=output)


generator = create_A0(40)
generator.load_weights("generator_weights_600.h5")

import chess
import numpy as np



def board_to_input_data(board: chess.Board) -> List[np.ndarray]:
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    piece_colors = [chess.WHITE, chess.BLACK]

    input_data = []

    for color in piece_colors:
        for piece_type in piece_types:
            # Create a binary mask for the specified piece type and color
            mask = np.zeros((8, 8),dtype=np.float32)
            for square in board.pieces(piece_type, color):
                mask[chess.square_rank(square)][chess.square_file(square)] = 1
            input_data.append(mask)
    input_data = np.transpose(input_data,(1,2,0))

    return np.array(input_data,dtype=np.float32)


dic_piece = {"P" : 0, "N" : 1, "B" : 2, "R" : 3, "Q" : 4, "K" : 5, "p" : 6, "n" : 7, "b" : 8, "r" : 9, "q" : 10, "k" : 11}


def board_to_transformer_input(board: chess.Board) -> np.ndarray:
    bitboard = np.zeros(32,dtype=np.int64)
    mask = np.zeros(32,dtype=np.int64)
    for i,piece in enumerate(board.piece_map()):
        #print(piece,board.piece_at(piece).symbol())
        bitboard[i] = piece+dic_piece[board.piece_at(piece).symbol()]*64
    mask = np.where(bitboard!=0,1,0)
    return bitboard,mask





board = chess.Board()
board.set_fen("r1b2r1k/1p1p1pbp/2n1pNp1/1p2P1B1/1P1P4/q4N2/P4PPP/R1Q1R1K1 b - - 4 15")

#set color to the side to play
if board.turn == chess.WHITE:
    color = 1
else:
    color = 0

x = []
Transformer_board= []
Transformer_mask = []
elo = 2200
elo = elo/3000
elo = np.ones((8,8),dtype=np.float32)*elo
elo = np.expand_dims(elo,axis=-1)
color = np.ones((8,8),dtype=np.float32)*color
color = np.expand_dims(color,axis=-1)


before = board_to_input_data(board)
Tboard,Tmask = board_to_transformer_input(board)
Transformer_board.append(Tboard)
Transformer_mask.append(Tmask)



Transformer_board = np.array(Transformer_board,dtype=np.int64)
Transformer_mask = np.array(Transformer_mask,dtype=np.int64)
x.append(np.concatenate((before,color,elo),axis=2))
x = np.array(x,dtype=np.float32)
x = [x,Transformer_board,Transformer_mask]

y = generator.predict(x)

print(y)


def get_move_from_ys(y,num_moves=10):
    #find the ten best moves
    y = np.argsort(y[0])
    y = y[::-1]
    #print the ten best moves
    moves = []
    
    for i in range(num_moves):
        ymax = y[i]
        #move_id = move.from_square + move.to_square*64
        from_square = ymax%64
        to_square = ymax//64
        from_square = chess.SQUARES[from_square]
        to_square = chess.SQUARES[to_square]
        move = chess.Move(from_square,to_square)
        moves.append(move)
    return moves

for move in get_move_from_ys(y):
    print(board)
    print(move.uci())