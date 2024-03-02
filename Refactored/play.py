import tensorflow as tf
import importlib
from config.string_to_elements import from_string_to_fun
from models.Leela_ResNet import create_model
from data_gen.gen_castling import policy_index,get_x_from_board,mirror_uci_string
import chess
import numpy as np
import time
import chess

from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget



model_weights = "/home/antoine/Bureau/projects/ParrotChess/model_2023-10-09_17-42-17_200.h5"
config_name = 'default_config'
def load_model(model_weights):
    
    converter = from_string_to_fun()
    
    params = importlib.import_module('config.'+ config_name).parameters()
    arguments = params['shared_parameters']
    training_args = params['training_args']
    data_generator = converter[params['data_generator']](arguments)
    arguments['num_channels'] = data_generator.out_channels
    model = converter[params['model']](arguments)
    model.load_weights(model_weights)
    return model






def predict(board,model, X):
    
    y_pred = model(X)
    y_prob = tf.keras.layers.Softmax()(y_pred)
    #tf.print(y_prob)
    #print(argmax)
    mirrored_board = board.mirror()
    legal_moves = [0]* len(policy_index)
    for move in mirrored_board.legal_moves:
        possible_str = move.uci()
        if possible_str[-1]!='n':
            legal_moves[policy_index.index(possible_str)] = 1
    legal_moves = tf.convert_to_tensor(np.array(legal_moves))
    legal_moves = tf.cast(legal_moves,tf.float32)
    argmax_before = tf.argmax(y_prob,axis=-1)[0]
    print(policy_index[argmax_before])
    y_prob = y_prob*legal_moves    
    argmax = tf.argmax(y_prob,axis=-1)[0]
    return policy_index[argmax]
    
   
   
   
   
def play(model,game_moves,elo):
    
    board = chess.Board()
    list_of_X = []
    
    for move in game_moves:
        
        list_of_X.append(get_x_from_board(elo,board))
        board.push(chess.Move.from_uci(move))
    list_of_X.append(get_x_from_board(elo,board))
    history = list_of_X[-8:]
            
    for i in range(7):
        history[i]=history[i][:,:,:12]
    
    X_final = np.concatenate(history,axis = -1)
    
    X_final = np.expand_dims(X_final,axis = 0)    
    
    return predict(board,model,X_final)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setGeometry(100, 100, 720, 720)

        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(10, 10, 720, 720)

        self.chessboard = chess.Board()

        self.chessboardSvg = chess.svg.board(self.chessboard, size = 250).encode("UTF-8")
        self.widgetSvg.load(self.chessboardSvg)
    def update(self, play_board):
        self.chessboard = play_board
        self.chessboardSvg = chess.svg.board(self.chessboard).encode("UTF-8")
        self.widgetSvg.load(self.chessboardSvg)
if __name__ == '__main__':
    
    model = load_model(model_weights)
    model.summary()
    elo = 2100
    play_board = chess.Board()
    game_moves = ["e2e4","e7e5","g1f3","b8c6","f1b5","a7a6","b5a4","g8f6","e1g1"]
    app = QApplication([])
    window = MainWindow()
    
    for move in game_moves:
        play_board.push(chess.Move.from_uci(move))
    while True:
        
        if len(game_moves)%2 ==1:
            move_string = mirror_uci_string(play(model,game_moves,elo))
        else:
            move_string =play(model,game_moves,elo)
        game_moves.append(move_string)
        print("AI chooses", move_string)
        play_board.push(chess.Move.from_uci(game_moves[-1]))
        window.update(play_board)
        window.show()
        app.exec()
        move = input("move ?")
        if move == "stop":
            break
        
        game_moves.append(move)
        print(game_moves)
        play_board.push(chess.Move.from_uci(game_moves[-1]))
    
    #print(play(model,game_moves,elo))
    
    





