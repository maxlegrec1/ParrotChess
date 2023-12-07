
import chess.pgn
import chess
import numpy as np
from typing import Tuple, List
import tensorflow as tf
from tqdm import tqdm
from scipy.stats import weibull_min
import random
from policy_index import get_policy_index
import ray
policy_index = get_policy_index()



def mirror_uci_string(uci_string):
    """
    Mirrors a uci string
    """
    if len(uci_string)<=4:
        return uci_string[0] + str(9 - int(uci_string[1])) + uci_string[2] + str(9 - int(uci_string[3]))
    else:
        return uci_string[0] + str(9 - int(uci_string[1])) + uci_string[2] + str(9 - int(uci_string[3])) + uci_string[4]





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



def get_board_data(pgn,board,real_move):
    if board.turn == chess.WHITE:
        elo = pgn.headers["WhiteElo"]
    else:
        elo = pgn.headers["BlackElo"]
    TC = pgn.headers['TimeControl']
    return get_board(elo,board,real_move,TC)

def get_board(elo,board,real_move,TC):
    if board.turn == chess.WHITE:
        color = 1
        mirrored_board = board.copy()
        move = real_move
    else:
        color = 0
        mirrored_board = board.mirror()
        mirror_uci = mirror_uci_string(real_move.uci())
        move = chess.Move.from_uci(mirror_uci)

    try:
        elo = float(elo)
    except:
        elo = 1500
    elo = elo/3000
    TC = TC.split('+')[0]
    if TC == '-':
        TC = 600
    else:
        TC = float(TC.split('+')[0])

    TC = TC / 120

    color = np.ones((8,8,1),dtype=np.float32)*color
    elo = np.ones((8,8,1),dtype=np.float32)*elo
    TC = np.ones((8,8,1),dtype=np.float32)*TC
    before = board_to_input_data(mirrored_board)

    #add castling rights for white and black
    castling_rights = np.ones((8,8,4),dtype=np.float32)
    if  not mirrored_board.has_kingside_castling_rights(chess.WHITE):
        castling_rights[:,:,0] = 0
    if not mirrored_board.has_queenside_castling_rights(chess.WHITE):
        castling_rights[:,:,1] = 0
    if not mirrored_board.has_kingside_castling_rights(chess.BLACK):
        castling_rights[:,:,2] = 0
    if not mirrored_board.has_queenside_castling_rights(chess.BLACK):
        castling_rights[:,:,3] = 0

    #add en passant rights
    en_passant_right = np.ones((8,8,1),dtype=np.float32)
    if not mirrored_board.has_pseudo_legal_en_passant():
        en_passant_right *= 0    

    lm =  - np.ones(1858,dtype=np.float32)
    for possible in mirrored_board.legal_moves:
        possible_str = possible.uci()
        if possible_str[-1]!='n':
            lm[policy_index.index(possible_str)] = 0



    #find the index of the move in policy_index

    move_id = policy_index.index(move.uci())

    one_hot_move = np.zeros(1858,dtype=np.float32)
    one_hot_move[move_id] = 1

    board.push(real_move)
    mirrored_board.push(move)

    one_hot_move = one_hot_move + lm


    return np.concatenate((before,castling_rights,en_passant_right,color,TC,elo),axis=2),one_hot_move
    
    
def load_game_RNN(pgn):
    if pgn.next() is None:
        print("NO MORE GAME TO LOAD, PGN DONE")
    if pgn.next()!=None:
        moves = [move for move in pgn.mainline_moves()]
        board = chess.Board()
        all_game_data = []
        for move in moves:
            x, y = get_board_data(pgn,board,move)
            all_game_data.append((x[:,:,:-2], np.concatenate([y, x[0,0,-2:]])))
        return all_game_data
    

def gen_dataset(nb_data, batch_size, nb_process):
    ray.init()
    in_pgn = "human2.pgn"
    nb_iterations = nb_data // (batch_size * nb_process)
    count = 0
    x_data = []
    y_data = []
    with open(in_pgn) as f:
        for iteration in range(nb_iterations):
            print(f"iteration {iteration} started")
            
            batches = [[chess.pgn.read_game(f) for _ in range(batch_size)] for _ in range(nb_process)]
            res = [ ray.remote(load_games).remote(batch) for batch in batches]
            data = ray.get(res)
            x = [data[i][0] for i in range(len(data))]
            y = [data[i][1] for i in range(len(data))]
            x_data.append(x)
            y_data.append(y)

def load_games(L_pgn):
    L = [load_game_RNN(pgn) for pgn in L_pgn]
    x = [L[i][0] for i in range(len(L))]
    y = [L[i][1] for i in range(len(L))]
    return x, y




    
if __name__ == "__main__":
    pgn_name = "human2.pgn"
    with open(pgn_name) as pgn:
        game = load_game_RNN(chess.pgn.read_game(pgn))
        print(game[0].shape,game[1].shape) 
