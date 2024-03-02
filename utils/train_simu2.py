import tensorflow as tf
import numpy as np
import chess

import chess.pgn

from data_gen.gen_TC_thinker import policy_index



channels = 12
def create_cnn3():
    x = tf.keras.Input((8,8,channels))

    y = tf.keras.Input((1858))


    y_reshaped = tf.keras.layers.Dense((8*8*1))(y)
    y_reshaped = tf.keras.layers.ReLU()(y_reshaped)
    y_reshaped = tf.keras.layers.BatchNormalization()(y_reshaped)
    y_reshaped = tf.keras.layers.Dropout(0.2)(y_reshaped)
    y_reshaped = tf.keras.layers.Reshape((8,8,1))(y_reshaped)


    def small_block():
        inputs = tf.keras.Input((8,8,channels+1))
        
        flow = tf.keras.layers.Conv2D(16,(9,9),padding="same")(inputs)
        flow = tf.keras.layers.ReLU()(flow)
        flow = tf.keras.layers.BatchNormalization()(flow)
        flow = tf.keras.layers.Dropout(0.2)(flow)
        flow = tf.keras.layers.Conv2D(16,(9,9),padding="same")(flow)
        flow = tf.keras.layers.ReLU()(flow)
        flow = tf.keras.layers.BatchNormalization()(flow)
        flow = tf.keras.layers.Dropout(0.2)(flow)
        res = tf.keras.layers.Conv2D(1,(9,9),padding="same")(flow)
        res = tf.keras.layers.Activation("sigmoid")(res)
        return tf.keras.Model(inputs = inputs, outputs = res)


    flow = tf.concat([small_block()(tf.concat([x,y_reshaped],axis=-1)) for i in range(channels)],axis=-1)


    

    
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
    if move.uci()[-1]=='n':
        move_id = policy_index.index(move.uci()[:-1])
    else:
        move_id = policy_index.index(move.uci())

    one_hot_move = np.zeros(1858,dtype=np.float32)
    one_hot_move[move_id] = 1

    return one_hot_move

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
                        
                    
                


'''
gen = generator_chess(256)
model = create_cnn3()
model.compile(optimizer='rmsprop',loss="binary_crossentropy",metrics = [myMetric,"mse"])
model.summary()
history = model.fit(gen,epochs = 20 ,steps_per_epoch=10000)


#save model under 'simulator.h5'
model.save('simulator.h5')'''

