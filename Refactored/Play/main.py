import sys
import os
cwd = os.getcwd()
sys.path.append(cwd)

from selenium import webdriver
from selenium.webdriver.common.by import By
import tensorflow as tf
from utils.pgn_to_input_data import list_uci_to_input
from Refactored.data_gen.gen_TC import policy_index, mirror_uci_string
from Refactored.models.BT4_LoRA import create_model
from Refactored.models.bt4_real import create_model as create_bt4
from utils.ustotheirs import ustotheirs
def update(MOVE_LIST,driver):
    i = len(MOVE_LIST)+1
    if i==1:
        string="//*[@id='main-wrap']/main/div[1]/rm6/l4x/kwdb"
    else:
        string="//*[@id='main-wrap']/main/div[1]/rm6/l4x/kwdb["+str(i)+"]"

    try:
        move_i = driver.find_element(By.XPATH,string)
        new_move = move_i.text.replace('½','').replace('?','').replace('#','')
        #print(move_i.text)
    except:
        new_move = None


    return new_move


def draw(move,driver):
    '''
    highlights the move_from square in green and move_to square in red by changing the html
    '''

    move = 'e2e4'

    move_from = move[:2]
    move_to = move[2:]

    # #ff000054



# This is the main file for the chess bot. It will be responsible for running the bot and the game.
#The architecture is going to be a state machine.

#defining the states
INIT = 0
WAITING_FOR_GAME = 1
PLAYING = 2
GAME_OVER = 3



STATES = [INIT, WAITING_FOR_GAME, PLAYING, GAME_OVER]




if __name__ == "__main__":
    CURR_STATE = INIT
    ELO = 3000
    MOVE_LIST = None
    GAME_ID = None
    #load model

    GPU_ID = 0
    tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[GPU_ID], 'GPU')
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[GPU_ID], True)

    model = create_model()
    model.load_weights("BT4_LoRA.h5")
    bt4 = create_bt4()
    #model.load_weights("F:\GitRefactored\ParrotChess\model_2024-02-10_11-07-25_160.h5")

    print(CURR_STATE)
    driver = webdriver.Chrome()

    driver.get("https://www.lichess.org")

    CURR_STATE = WAITING_FOR_GAME

    print(CURR_STATE)
    while True:
        #print("Current State : ", CURR_STATE)

        if CURR_STATE == WAITING_FOR_GAME:
            #print(driver.current_url.split("/")[-1]) 

            if len(driver.current_url.split("/")[-1]) >=5:
                print("Game Found")
                CURR_STATE = PLAYING
                GAME_ID = driver.current_url.split("/")[-1]
                MOVE_LIST = []


        if CURR_STATE == PLAYING:

            #print(MOVE_LIST)
            new_move = update(MOVE_LIST,driver)
            if new_move != None:
                MOVE_LIST.append(new_move)
                print(MOVE_LIST) 
            if len(MOVE_LIST)>=8 and new_move!=None:
                X,mask = list_uci_to_input(MOVE_LIST,ELO,"300")
                #do the inference here
                Y = model(X)
                #print(ustotheirs(X[0])[0,108])
                Y_bt4 = bt4(ustotheirs(X[0]))
                best = tf.argmax(0.60*(mask+Y_bt4['policy'])+0.4*Y,axis=-1)
                move_id = best.numpy()[0]
                move = policy_index[move_id]
                if len(MOVE_LIST)%2 == 1:
                    print("black")
                    move = mirror_uci_string(move)
                print(move,Y_bt4['value_winner'].numpy())

            if len(driver.current_url.split("/")[-1]) <=5:
                print("Back to the main page")
                CURR_STATE = WAITING_FOR_GAME
                GAME_ID = None
                MOVE_LIST = None    
            try:
                driver.find_element(By.CLASS_NAME,"result")
                print("Game Over")
                CURR_STATE = GAME_OVER
                MOVE_LIST = None
            except:
                pass

            


        if CURR_STATE == GAME_OVER:
            

            if driver.current_url.split("/")[-1] != GAME_ID:
                print(driver.current_url)
                print(GAME_ID)
                print("New Game Detected")
                CURR_STATE = PLAYING
                GAME_ID = driver.current_url.split("/")[-1]
                MOVE_LIST = []

            if len(driver.current_url.split("/")[-1]) <=5:
                print("Back to the main page")
                CURR_STATE = WAITING_FOR_GAME
                GAME_ID = None
                MOVE_LIST = None