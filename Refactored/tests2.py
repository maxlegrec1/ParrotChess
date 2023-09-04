import chess
import numpy as np
#board = chess.Board()
#board.set_fen("r1bqkb1r/ppp2ppp/2p2n2/4p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5")

#dic_piece = {"p": 0 , "n": 1, "b": 2, "r": 3, "q": 4, "k": 5, "P": 6, "N": 7, "B": 8, "R": 9, "Q": 10, "K": 11}
dic_piece = {"P" : 0, "N" : 1, "B" : 2, "R" : 3, "Q" : 4, "K" : 5, "p" : 6, "n" : 7, "b" : 8, "r" : 9, "q" : 10, "k" : 11}


def board_to_transformer_input(board: chess.Board) -> np.ndarray:
    bitboard = np.zeros(32,dtype=np.int64)
    mask = np.zeros((32,32),dtype=np.int64)
    for i,piece in enumerate(board.piece_map()):
        #print(piece,board.piece_at(piece).symbol())
        bitboard[i] = piece+dic_piece[board.piece_at(piece).symbol()]*64
    mask_length = np.sum(np.where(bitboard!=0,1,0))
    mask[:mask_length,:mask_length] = 1
    return bitboard,mask

#print(board_to_transformer_input(board))