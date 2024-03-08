import sys
import os
import numpy as np
import time
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
cwd = os.getcwd()
sys.path.append(cwd)

from Refactored.models.BT4 import create_model
from utils.pgn_to_input_data import list_uci_to_input
from Refactored.data_gen.gen_TC import data_gen

params = {'batch_size': 1,'path_pgn': 'human2.pgn'}
gen = data_gen(params)
model = create_model()
#create a csv 'results.csv'
with open('results.csv', 'w') as f:
    f.write('pos_id,elo,tc,is_white,move_number,has_found\n')


    NUM_POS = 1_0

    for i in tqdm(range(NUM_POS)):

        X,Y_true = gen.get_batch()
        move_number = X[0][0,0,0,-2]
        if move_number<0:#not en passant
            X[0][:,:,:,-2]*=0
        else:
            X[0][:,:,:,-2]= 1

        Y = model(X)

        elo = X[1][0,0,0,1]*3000
        tc = X[1][0,0,0,0]*120
        color = X[0][0,0,0,-1]
        


        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y_true,axis=-1),tf.argmax(Y,axis=-1)),tf.float32),axis=-1).numpy()
        #add a row in the dataframe
        f.write(f'{i},{elo},{tc},{color},{move_number},{acc}\n')




