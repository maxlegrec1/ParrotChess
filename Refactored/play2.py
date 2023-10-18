# importing required librarys
import pygame
import chess
import math
import tensorflow as tf
import importlib
from config.string_to_elements import from_string_to_fun
from data_gen.gen_castling import policy_index,get_x_from_board,mirror_uci_string
import chess
import numpy as np
import time
import chess

#initialise display
X = 512
x = X // 8
Y = 512
y = Y // 8
scrn = pygame.display.set_mode((X, Y))
pygame.init()

#basic colours
WHITE = (255, 255, 255)
GREY = (128, 128, 128)
YELLOW = (204, 204, 0)
BLUE = (50, 255, 255)
BLACK = (0, 0, 0)

#initialise chess board
b = chess.Board()

#load piece images
pieces = {'P': pygame.image.load('/home/antoine/Bureau/projects/ParrotChess/img/Chess_plt45.svg.png').convert_alpha(),
          'N': pygame.image.load('/home/antoine/Bureau/projects/ParrotChess/img/Chess_nlt45.svg.png').convert_alpha(),
          'B': pygame.image.load('/home/antoine/Bureau/projects/ParrotChess/img/Chess_blt45.svg.png').convert_alpha(),
          'R': pygame.image.load('/home/antoine/Bureau/projects/ParrotChess/img/Chess_rlt45.svg.png').convert_alpha(),
          'Q': pygame.image.load('/home/antoine/Bureau/projects/ParrotChess/img/Chess_qlt45.svg.png').convert_alpha(),
          'K': pygame.image.load('/home/antoine/Bureau/projects/ParrotChess/img/Chess_klt45.svg.png').convert_alpha(),
          'p': pygame.image.load('/home/antoine/Bureau/projects/ParrotChess/img/Chess_pdt45.svg.png').convert_alpha(),
          'n': pygame.image.load('/home/antoine/Bureau/projects/ParrotChess/img/Chess_ndt45.svg.png').convert_alpha(),
          'b': pygame.image.load('/home/antoine/Bureau/projects/ParrotChess/img/Chess_bdt45.svg.png').convert_alpha(),
          'r': pygame.image.load('/home/antoine/Bureau/projects/ParrotChess/img/Chess_rdt45.svg.png').convert_alpha(),
          'q': pygame.image.load('/home/antoine/Bureau/projects/ParrotChess/img/Chess_qdt45.svg.png').convert_alpha(),
          'k': pygame.image.load('/home/antoine/Bureau/projects/ParrotChess/img/Chess_kdt45.svg.png').convert_alpha(),
          
          }

def update(scrn,board):
    '''
    updates the screen basis the board class
    '''
    
    for i in range(64):
        piece = board.piece_at(i)
        if piece == None:
            pass
        else:
            scrn.blit(pieces[str(piece)],((i%8) * x+8,X-x -(i//8)*x+8))
    
    for i in range(7):
        i=i+1
        pygame.draw.line(scrn,WHITE,(0,i*x),(X,i*x))
        pygame.draw.line(scrn,WHITE,(i*x,0),(i*x,X))

    pygame.display.flip()
def main(BOARD):

    '''
    for human vs human game
    '''
    #make background black
    scrn.fill(GREY)
    #name window
    pygame.display.set_caption('Chess')
    
    #variable to be used later
    index_moves = []

    status = True
    while (status):
        #update screen
        update(scrn,BOARD)

        for event in pygame.event.get():
     
            # if event object type is QUIT
            # then quitting the pygame
            # and program both.
            if event.type == pygame.QUIT:
                status = False

            # if mouse clicked
            if event.type == pygame.MOUSEBUTTONDOWN:
                #remove previous highlights
                scrn.fill(GREY)
                #get position of mouse
                pos = pygame.mouse.get_pos()

                #find which square was clicked and index of it
                square = (math.floor(pos[0]/x),math.floor(pos[1]/x))
                index = (7-square[1])*8+(square[0])
                
                # if we are moving a piece
                if index in index_moves: 
                    
                    move = moves[index_moves.index(index)]
                    
                    BOARD.push(move)

                    #reset index and moves
                    index=None
                    index_moves = []
                    
                    
                # show possible moves
                else:
                    #check the square that is clicked
                    piece = BOARD.piece_at(index)
                    #if empty pass
                    if piece == None:
                        
                        pass
                    else:
                        
                        #figure out what moves this piece can make
                        all_moves = list(BOARD.legal_moves)
                        moves = []
                        for m in all_moves:
                            if m.from_square == index:
                                
                                moves.append(m)

                                t = m.to_square

                                TX1 = x*(t%8)
                                TY1 = x*(7-t//8)

                                
                                #highlight squares it can move to
                                pygame.draw.rect(scrn,BLUE,pygame.Rect(TX1,TY1,x,x),5)
                        
                        index_moves = [a.to_square for a in moves]
     
    # deactivates the pygame library
        if BOARD.outcome() != None:
            print(BOARD.outcome())
            status = False
            print(BOARD)
    pygame.quit()

def main_one_agent(BOARD, game_moves, model,agent_color, elo):
    
    '''
    for agent vs human game
    color is True = White agent
    color is False = Black agent
    '''
    
    #make background black
    scrn.fill(GREY)
    #name window
    pygame.display.set_caption('Chess')
    
    #variable to be used later
    index_moves = []
    status = True

    while (status):
        #update screen
        update(scrn,BOARD)
        
     
        if BOARD.turn==agent_color:
            if len(game_moves)%2 ==1:
                move_string = mirror_uci_string(play(model,game_moves,elo))
            else:
                move_string =play(model,game_moves,elo)
            game_moves.append(move_string)
            play_board.push(chess.Move.from_uci(game_moves[-1]))
            print("AI chooses", move_string)
            scrn.fill(GREY)

        else:

            for event in pygame.event.get():
         
                # if event object type is QUIT
                # then quitting the pygame
                # and program both.
                if event.type == pygame.QUIT:
                    status = False

                # if mouse clicked
                if event.type == pygame.MOUSEBUTTONDOWN:
                    #reset previous screen from clicks
                    scrn.fill(GREY)
                    #get position of mouse
                    pos = pygame.mouse.get_pos()

                    #find which square was clicked and index of it
                    square = (math.floor(pos[0]/x),math.floor(pos[1]/x))
                    index = (7-square[1])*8+(square[0])
                    
                    # if we have already highlighted moves and are making a move
                    if index in index_moves: 
                        
                        move = moves[index_moves.index(index)]
                        #print(BOARD)
                        game_moves.append(str(move))
                        play_board.push(chess.Move.from_uci(game_moves[-1]))
                        index=None
                        index_moves = []
                        
                    # show possible moves
                    else:
                        
                        piece = BOARD.piece_at(index)
                        
                        if piece == None:
                            
                            pass
                        else:

                            all_moves = list(BOARD.legal_moves)
                            moves = []
                            for m in all_moves:
                                if m.from_square == index:
                                    
                                    moves.append(m)

                                    t = m.to_square

                                    TX1 = x*(t%8)
                                    TY1 = x*(7-t//8)

                                    
                                    pygame.draw.rect(scrn,BLUE,pygame.Rect(TX1,TY1,x,x),5)
                            #print(moves)
                            index_moves = [a.to_square for a in moves]
     
    # deactivates the pygame library
        if BOARD.outcome() != None:
            print(BOARD.outcome())
            status = False
            print(BOARD)
    pygame.quit()




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
    
    X_final_no_elo = X_final[:,:,:,:X_final.shape[-1]-1]
    elo = X_final[:,:,:,X_final.shape[-1]-1:]
    
    return predict(board,model,[X_final_no_elo, elo])

model_weights = "/home/antoine/Bureau/projects/ParrotChess/model_2023-10-18_19-36-03_320.h5"
config_name = 'default_config'

if __name__ == '__main__':
    
    model = load_model(model_weights)
    model.summary()
    elo = 3000
    play_board = chess.Board()
    game_moves = ["d2d4","d7d5","b1c3","g8f6","c1f4","e7e6","c3b5","c8d7","b5c7"]
    
    for move in game_moves:
        play_board.push(chess.Move.from_uci(move))
    main_one_agent(play_board,game_moves, model, False, elo)
        
