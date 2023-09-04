
import chess.pgn
import chess
import numpy as np
from typing import Tuple, List
import tensorflow as tf

def generate_legalmove_mask(board):
    #mask has for shape (8,8,64)
    mask = np.zeros((8,8,64),dtype=np.float32)
    for move in board.generate_legal_moves():
        mask[chess.square_rank(move.from_square),chess.square_file(move.from_square),8*chess.square_rank(move.to_square)+chess.square_file(move.to_square)] = 1
        #print(move,(chess.square_rank(move.from_square),chess.square_file(move.from_square),8*chess.square_rank(move.to_square)+chess.square_file(move.to_square)))

    return mask


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

    return np.array(input_data,dtype=np.float32)



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


def generate_batch():
    batch_size = 32
    total_games = 0
    total_pos = 0
    in_pgn = "human.pgn"
    out_csv = "human.csv"
    y = []
    x = []
    y_prob = []
    disc_true = []
    with open(in_pgn) as f:
        while True:
            pgn = chess.pgn.read_game(f)
            if pgn.next()!=None:
                moves = [move for move in pgn.mainline_moves()]
                if len(moves)>=10:
                    total_games+=1
                    #make the 10 first moves 
                    board = chess.Board()
                    for move in moves[:10]:
                        board.push(move)
                    for move in moves[10:]:
                        if total_pos%batch_size==0 and total_pos!=0:
                            x = np.array(x,dtype=np.float32)
                            y_prob = np.array(y_prob,dtype=np.float32)
                            noise = np.random.normal(0, 1, (x.shape[0],8,8,1)).astype(np.float32)
                            x_prim = np.concatenate((x,noise),axis=3)
                            #yield (x_prim,np.array(y,dtype=np.float32),x)
                            yield (x,y_prob)
                            x = []
                            y = []
                            y_prob = []
                            disc_true = []
                        total_pos+=1
                        if board.turn == chess.WHITE:
                            color = 1
                            elo = pgn.headers["WhiteElo"]
                        else:
                            color = 0
                            elo = pgn.headers["BlackElo"]
                        elo = float(elo)
                        elo = elo/3000
                        color = np.ones((8,8),dtype=np.float32)*color
                        #add one channel
                        color = np.expand_dims(color,axis=-1)
                        elo = np.ones((8,8),dtype=np.float32)*elo
                        elo = np.expand_dims(elo,axis=-1)
                        legal_mask = generate_legalmove_mask(board)
                        before,after = board_and_move_to_input_data(board,move)
                        #print(move.uci,move.from_square,move.to_square)
                        move_id = move.from_square + move.to_square*64
                        #one hot encode the move
                        one_hot_move = np.zeros(4096,dtype=np.float32)
                        one_hot_move[move_id] = 1
                        y_prob.append(one_hot_move)
                        #x.append(np.concatenate((before,legal_mask,color,elo),axis=2))
                        x.append(np.concatenate((before,color,elo),axis=2))
                        y.append(after)
                        disc_true.append(before)
                        #data.append(np.concatenate((before,legal_mask,color,elo,after),axis=2))
                        #print the number of pos per game
                        #print(total_pos/total_games)

#gen = generate_batch()
#x = next(gen)

