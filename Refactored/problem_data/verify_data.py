import berserk
import pandas as pd
import tqdm
import time
import multiprocessing
from multiprocessing import Process
import chess
import chess.pgn
multiprocessing.set_start_method('fork')

PATH_PB = "Refactored/problem_data/problem.pgn"
PATH_INFO = "Refactored/problem_data/info_problem.pgn"

            

def get_id_pgn(pgn):
    id_pgn = pgn.headers['Site'].split('/')[-1]
    assert len(id_pgn) == 8, f'Invalid parse id'
    return id_pgn

def get_id_pb(pb):
    id_pb = pb.split('/')[1]
    assert len(id_pb) == 8, f'Invalid parse id'
    return id_pb
def check_consistency():
    with open(PATH_PB, 'r') as file_pb:
        with open(PATH_INFO, 'r') as file_info:
            i=0
            while True:
                if i % 10000 == 0:
                    print(i)
                pgn =chess.pgn.read_game(file_pb)
                problem_moves = file_info.readline()
                if pgn is None or problem_moves is None:
                    assert pgn is not None or problem_moves is not None, f'problem not same length at {i}'
                    break
                if get_id_pb(problem_moves) != get_id_pgn(pgn):
                    assert f'id not matching at line {i}'
                i+=1
                
                
def test_number_game_in_pgn():
    with open(PATH_PB) as f:
        i=0
        while True:
            if i % 10000 == 0:
                print(i)
            pgn=chess.pgn.read_game(f)
            if pgn is None:
                break
            i+=1
            
    print(f'nb pb in pgn : {i}')
def test_number_pb_in_info():
    with open(PATH_INFO, 'r') as file_info:
        problem_moves = file_info.readlines()
        print(f'nb lines : {len(problem_moves)}, last elt : {problem_moves[-1]}')
    
if __name__ == '__main__':
    test_number_game_in_pgn()
    test_number_pb_in_info()
    # check_consistency()