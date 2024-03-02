
import chess
import chess.pgn

import os
import sys
import numpy as np
import time

cwd = os.getcwd()
sys.path.append(cwd)

from Refactored.data_gen.gen_TC import get_x_from_board,swap_side

def list_uci_to_input(list_of_moves,elo,TC):
    board = chess.Board()
    color = len(list_of_moves)

    X = []


    for i,m in enumerate(list_of_moves):
        real_move = board.parse_san(m)
        if i==len(list_of_moves)-1:
            print("geto",board.turn == chess.WHITE)
        xs = get_x_from_board(elo,board,TC)
        board.push(real_move)
        if (color-i)%2 == 1:
            X.append(swap_side(xs[:,:,:12]))
        else:
            X.append(xs[:,:,:12])

    #add last position
    X.append(get_x_from_board(elo,board,TC))




    X_final = np.concatenate([x[:,:,:12] for x in X[-8:-1]], axis = -1)
    X_final = np.concatenate([X_final,X[-1]], axis = -1)


    return [np.expand_dims(X_final[:,:,:-2],axis = 0),np.expand_dims(X_final[:,:,-2:],axis=0)]


if __name__=="__main__":
    game = ['Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8']
    print(len(game))
    t_0 = time.time()
    a,b = list_uci_to_input(game, 2000, '300')
    print(time.time()-t_0)
    print(a.shape,b.shape)