
import chess.pgn
import chess
import numpy as np
from typing import Tuple, List
import tensorflow as tf
from tqdm import tqdm

dic_piece = {"P" : 0, "N" : 1, "B" : 2, "R" : 3, "Q" : 4, "K" : 5, "p" : 6, "n" : 7, "b" : 8, "r" : 9, "q" : 10, "k" : 11}


def board_to_transformer_input(board: chess.Board) -> np.ndarray:

    if board.turn == chess.WHITE:
        color = 1
    else:
        color = 0

    bitboard = np.zeros(33,dtype=np.int64)
    bitboard[0] = 12*64 + color
    mask = np.zeros((33,33),dtype=np.int64)
    for i,piece in enumerate(board.piece_map()):
        #print(piece,board.piece_at(piece).symbol())
        bitboard[i+1] = piece+dic_piece[board.piece_at(piece).symbol()]*64
    mask_length = np.sum(np.where(bitboard!=0,1,0))
    mask[:mask_length,:mask_length] = 1
    return bitboard,mask



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


def generate_batch(batch_size,in_pgn,use_transformer = True, use_only_transformer = False):
    total_pos = 0
    x = []
    y_true = []
    Transformer_board= []
    Transformer_mask = []
    Legal_moves = []
    with open(in_pgn) as f:
        while True:
            #load game
            pgn = chess.pgn.read_game(f)
            if pgn.next()!=None:
                moves = [move for move in pgn.mainline_moves()]
                if len(moves)>=10:
                    #make the 10 first moves 
                    board = chess.Board()
                    for move in moves[:10]:
                        board.push(move)
                    for move in moves[10:]:
                        if total_pos%batch_size==0 and total_pos!=0:
                            x = np.array(x,dtype=np.float32)
                            y_true = np.array(y_true,dtype=np.float32)
                            Legal_moves = np.array(Legal_moves,dtype=np.float32)
                            Transformer_board = np.array(Transformer_board,dtype=np.int64)
                            Transformer_mask = np.array(Transformer_mask,dtype=np.int64)
                            if use_transformer:
                                x = [x,Transformer_board,Transformer_mask]
                            if use_only_transformer:
                                x = [Transformer_board,Transformer_mask]
                            else:
                                x = x
                            yield (x,y_true,Legal_moves)

                            #reset variables
                            Legal_moves = []
                            x = []
                            Transformer_board = []
                            Transformer_mask = []
                            y_true = []

                        if use_transformer:
                            xs,ys,Legal_move,Tboard,Tmask = get_board_data(pgn,board,move,use_transformer)

                        else:
                            xs,ys,Legal_move = get_board_data(pgn,board,move,use_transformer)

                        x.append(xs)
                        y_true.append(ys)
                        Legal_moves.append(Legal_move)
                        if use_transformer:
                            Transformer_board.append(Tboard)
                            Transformer_mask.append(Tmask)

                        total_pos+=1



def get_board_data(pgn,board,move,use_transformer = True):
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
    color = np.ones((8,8,1),dtype=np.float32)*color
    elo = np.ones((8,8,1),dtype=np.float32)*elo
    before = board_to_input_data(board)
    lm = np.zeros(4096,dtype=np.float32)
    for possible in board.legal_moves:
        lm[possible.from_square + possible.to_square*64] = 1
    if use_transformer:
        Tboard,Tmask = board_to_transformer_input(board)
    board.push(move)
    move_id = move.from_square + move.to_square*64
    one_hot_move = np.zeros(4096,dtype=np.float32)
    one_hot_move[move_id] = 1
    if use_transformer:
        return np.concatenate((before,color,elo),axis=2),one_hot_move,lm,Tboard,Tmask
    else:
        return np.concatenate((before,color,elo),axis=2),one_hot_move,lm


