import tensorflow as tf
import numpy as np
import chess

import chess.pgn

from data_gen.gen_TC_thinker import policy_index



channels = 12
def create_cnn3():
    x = tf.keras.Input((8,8,channels))

    y = tf.keras.Input((8,8,4))

    def small_block():
        inputs = tf.keras.Input((8,8,channels+4))
        
        flow = tf.keras.layers.Conv2D(16,(9,9),padding="same")(inputs)
        flow = tf.keras.layers.ReLU()(flow)
        flow = tf.keras.layers.BatchNormalization()(flow)
        flow = tf.keras.layers.Conv2D(16,(9,9),padding="same")(flow)
        flow = tf.keras.layers.ReLU()(flow)
        flow = tf.keras.layers.BatchNormalization()(flow)
        res = tf.keras.layers.Conv2D(1,(9,9),padding="same")(flow)
        res = tf.keras.layers.Activation("sigmoid")(res)
        return tf.keras.Model(inputs = inputs, outputs = res)


    flow = tf.concat([small_block()(tf.concat([x,y],axis=-1)) for i in range(channels)],axis=-1)


    

    
    output = flow
    
    return tf.keras.Model(inputs = [x,y], outputs = output)



def mirror_uci_string(uci_string):
    """
    Mirrors a uci string
    """
    if len(uci_string)<=4:
        return uci_string[0] + str(9 - int(uci_string[1])) + uci_string[2] + str(9 - int(uci_string[3]))
    else:
        return uci_string[0] + str(9 - int(uci_string[1])) + uci_string[2] + str(9 - int(uci_string[3])) + uci_string[4]



from typing import List
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

def move_to_input_data(move:chess.Move) -> List[np.ndarray]:
    input_data = np.zeros((8,8,4),dtype = np.float32)
    f = move.from_square
    input_data[chess.square_rank(f),chess.square_file(f),0] = -1
    t = move.to_square
    input_data[chess.square_rank(t),chess.square_file(t),0] = 1
    if move.uci()[-1]=='q':
        input_data[:,:,1] = 1
    elif move.uci()[-1]=='r':
        input_data[:,:,2] = 1
    elif move.uci()[-1]=='b':
        input_data[:,:,3] = 1
    return input_data

def myMetric(y_true,y_pred):
    processed = tf.greater(y_pred,0.5)
    processed = tf.cast(processed,tf.float32)
    diff = tf.abs(y_true-processed)
    loss = tf.reduce_mean(diff,[1,2,3])
    return loss



def generator_chess(batch_size):
    x_0 = []
    x_1 = []
    y = []
    with open("human2.pgn") as f:
        while True:
            pgn = chess.pgn.read_game(f)
            if pgn.next()!= None:
                moves = [move for move in pgn.mainline_moves()]
                board = chess.Board()
                for move in moves:
                    if len(x_0)==batch_size:
                        x_0 = np.array(x_0,dtype = np.float32)
                        x_1 = np.array(x_1,dtype = np.float32)
                        y = np.array(y,dtype= np.float32)
                        yield ([x_0,x_1],y)
                        x_0 = []
                        x_1 = []
                        y = []
                    if board.turn == chess.WHITE:    
                        x_0.append(board_to_input_data(board))
                        x_1.append(move_to_input_data(move))
                        board.push(move)
                        y.append(board_to_input_data(board))
                    else:
                        x_0.append(board_to_input_data(board.mirror()))
                        x_1.append(move_to_input_data(chess.Move.from_uci(mirror_uci_string(move.uci()))))
                        board.push(move)
                        y.append(board_to_input_data(board.mirror()))
                        
                    
                


gen = generator_chess(256)
model = create_cnn3()
model.compile(optimizer='rmsprop',loss="binary_crossentropy",metrics = [myMetric,"mse"])
model.summary()
history = model.fit(gen,epochs = 100 ,steps_per_epoch=10000)



