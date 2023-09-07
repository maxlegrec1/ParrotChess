
import chess.pgn
import chess
import numpy as np
from typing import Tuple, List
import tensorflow as tf
from tests2 import board_to_transformer_input
from tqdm import tqdm

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


def generate_batch():
    prout = []
    batch_size = 32
    total_games = 0
    total_pos = 0
    in_pgn = "human.pgn"
    x = []
    y_prob = []
    Transformer_board= []
    Transformer_mask = []
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
                    #last move
                    last = moves[9]
                    for move in moves[10:]:
                        if total_pos%batch_size==0 and total_pos!=0:
                            x = np.array(x,dtype=np.float32)
                            y_prob = np.array(y_prob,dtype=np.float32)
                            prout = np.array(prout,dtype=np.float32)
                            Transformer_board = np.array(Transformer_board,dtype=np.int64)
                            Transformer_mask = np.array(Transformer_mask,dtype=np.int64)
                            x = [x,Transformer_board,Transformer_mask]
                            yield (x,y_prob,prout)
                            prout = []
                            x = []
                            Transformer_board = []
                            Transformer_mask = []
                            y_prob = []
                        total_pos+=1
                        if board.turn == chess.WHITE:
                            color = 1
                            elo = pgn.headers["WhiteElo"]
                        else:
                            color = 0
                            elo = pgn.headers["BlackElo"]
                        try:
                            elo = float(elo)
                        except:
                            elo = 1500
                        elo = elo/3000
                        color = np.ones((8,8),dtype=np.float32)*color
                        #add one channel
                        color = np.expand_dims(color,axis=-1)
                        elo = np.ones((8,8),dtype=np.float32)*elo
                        elo = np.expand_dims(elo,axis=-1)
                        before = board_to_input_data(board)
                        lm = np.zeros(4096,dtype=np.float32)
                        for possible in board.legal_moves:
                            lm[possible.from_square + possible.to_square*64] = 1
                        prout.append(lm)
                        Tboard,Tmask = board_to_transformer_input(board)
                        Transformer_board.append(Tboard)
                        Transformer_mask.append(Tmask)


                        board.push(move)
                        #print(move.uci,move.from_square,move.to_square)
                        move_id = move.from_square + move.to_square*64
                        #one hot encode the ys
                        one_hot_move = np.zeros(4096,dtype=np.float32)
                        one_hot_move[move_id] = 1
                        y_prob.append(one_hot_move)

                        #x.append(np.concatenate((before,legal_mask,color,elo),axis=2))
                        x.append(np.concatenate((before,color,elo),axis=2))


gen = generate_batch()

for _ in tqdm(range(1000)):
    next(gen)

