

import chess.pgn
import chess
import numpy as np
from typing import Tuple, List





def generate_legalmove_mask(board):
    #mask has for shape (8,8,64)
    mask = np.zeros((8,8,64))
    for move in board.generate_legal_moves():
        mask[chess.square_rank(move.from_square),chess.square_file(move.from_square),8*chess.square_rank(move.to_square)+chess.square_file(move.to_square)] = 1
        print(move,(chess.square_rank(move.from_square),chess.square_file(move.from_square),8*chess.square_rank(move.to_square)+chess.square_file(move.to_square)))

    return mask

'''
board = chess.Board()
board.set_fen("8/1k3P2/8/8/8/8/8/3K4 w - - 0 1")

print(generate_legalmove_mask(board).shape)'''


#load generator from file checkpoints/generator_12000.h5

import tensorflow as tf
from gan import create_A0
from gan import create_discriminator
discriminator = create_discriminator((8,8,90))
generator = create_A0(40,12)
#load the weights
generator.load_weights("checkpoints/generator_16000.h5")
discriminator.load_weights("checkpoints/discriminator_16000.h5")
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
from refactored import generate_batch
'''
gen = generate_batch()
x,y,disc_true = next(gen)'''


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
    input_data = np.array(input_data,dtype=np.float32)
    input_data = np.transpose(input_data,(1,2,0))
    return input_data



def board_and_move_to_input_data(board: chess.Board, move: chess.Move) -> np.ndarray:
    # Get the input data for the board before the move
    input_data_before = board_to_input_data(board)
    fen = board.fen()
    # Apply the move to the board
    board.push(move)

    # Get the input data for the board after the move
    input_data_after = board_to_input_data(board)

    # Concatenate the two input data arrays
    input_data = np.concatenate((input_data_before, input_data_after), axis=0)

    #convert to HWC
    input_data_before = np.transpose(input_data_before,(1,2,0))
    input_data_after = np.transpose(input_data_after,(1,2,0))
    return input_data_before,input_data_after




def generate_x_from_board(board):
    if board.turn == chess.WHITE:
        color = 1
        elo = 1600
    else:
        color = 0
        elo = 1600
    elo = float(elo)
    color = np.ones((8,8),dtype=np.float32)*color
    #add one channel
    color = np.expand_dims(color,axis=-1)
    elo = np.ones((8,8),dtype=np.float32)*elo
    elo = np.expand_dims(elo,axis=-1)
    legal_mask = generate_legalmove_mask(board)
    before = board_to_input_data(board)
    x=[]
    x.append(np.concatenate((before,legal_mask,color,elo),axis=2))
    x = np.array(x,dtype=np.float32)
    noise = np.random.normal(0, 1, (x.shape[0],8,8,1)).astype(np.float32)
    x = np.concatenate((x,noise),axis=3)
    return x


board = chess.Board()
board.push_san("e4")
x  = generate_x_from_board(board)
#remove last channel from x in y
y = x[:,:,:,:78]
gen_outputs = generator(x, training=False).numpy()

gen_outputs = np.transpose(gen_outputs,(0,3,1,2))
print(gen_outputs[0])
#fake_input_disc = tf.concat([y, gen_outputs], axis=-1)
#fake_output = discriminator(fake_input_disc, training=False)
#print(fake_output)
print('------------------------')
print(np.transpose(board_to_input_data(board),(2,0,1)))