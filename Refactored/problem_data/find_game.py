import berserk
import pandas as pd
import tqdm
import time
import multiprocessing
from multiprocessing import Process
import chess
import chess.pgn
multiprocessing.set_start_method('fork')
data = pd.read_csv('/home/antoine/Bureau/projects/ParrotChess/Refactored/problem_data/lichess_db_puzzle.csv')

Rating = data['Rating']
Moves = data['Moves']
Url = data['GameUrl']
args = {'evals': False, 'clocks': False, 'as_pgn':True}
def parse_fen(fen):
    L=[]
    for elt in fen:
        move_number = elt.split(' ')[-1]
        L.append(move_number)
    return L
def parse_url(url):
    L_url=[]
    L_number_move = []
    L_move_pb = []
    for index, elt in enumerate(url):
        end_url = elt.split('.org/')[-1]
        splitted_end = end_url.split('/')
        if len(splitted_end)!=1:
            L_url.append(splitted_end[0])
            assert len(splitted_end[0]) == 8
            number_move = splitted_end[1].split('#')[-1]
            assert int(number_move) %2 ==0
            L_number_move.append(number_move)
            L_move_pb.append(Moves[index])
        else:
            ashtag_split = splitted_end[0].split('#')
            assert len(ashtag_split[0]) == 8
            number_move = ashtag_split[1]
            assert int(number_move) % 2 ==1
            L_url.append(ashtag_split[0])
            L_number_move.append(number_move)
            L_move_pb.append(Moves[index])
    return L_url,L_number_move, L_move_pb

def example():
    with open('/home/antoine/Bureau/projects/ParrotChess/Refactored/problem_data/lichess.token') as f:
        token = f.read()
    session = berserk.TokenSession(token)
    client = berserk.Client(session)
    example_url = '787zsVup'
    example_url2 = 'MQSyb3KW'

    game1 = client.games.export(example_url, **args)
    game2 = client.games.export(example_url2, **args)
    
def not_used():
    with open('/home/antoine/Bureau/projects/ParrotChess/Refactored/problem_data/lichess.token') as f:
        token = f.read()
    session = berserk.TokenSession(token)
    client = berserk.Client(session)
    Moves = []
    file_pgn = open("problem.pgn", "a")
    file_problem_data = open("problem.info", "a")
    L_url,L_number_move = parse_url(Url)
    nb_problems = len(L_url)
    assert nb_problems == len(L_number_move)
    assert nb_problems == len(Moves)
    for i in tqdm.tqdm(range(nb_problems)):
        file_pgn.write(client.games.export(L_url[i], **args))
        file_problem_data.write(L_number_move[i] + ',' + Moves[i])
        time.sleep(0.01)
    file_pgn.close()
    file_problem_data.close()

def create_dict():
    L_url,L_number_move, L_move_pb = parse_url(Url)
    nb_problems = len(L_url)
    assert nb_problems == len(L_number_move)
    
    for elt in L_url:
        assert len(elt)==8
    return {elt : (str(L_number_move[index]), str(L_move_pb[index])) for index,elt in enumerate(L_url)}


def keep_pgn_problem(path_file_pgn, dico):
    
    pgn_problem = open("Refactored/problem_data/problem.pgn", "a")
    info_problem = open("Refactored/problem_data/info_problem.pgn", "a")
    i = 0
    m=0
    with open(path_file_pgn) as f:
        while True:
            if m*10000< i:
                print(m*10000)
                m+=1
            pgn = chess.pgn.read_game(f)
            if pgn is None:
                print(i)
                break
            if pgn.next() is not None:
                id_pgn = pgn.headers['Site'].split('/')[-1]
                assert len(id_pgn) == 8, f'Invalid parse id {i}'
                
                if id_pgn in dico:
                    pgn_problem.write(str(pgn) + "\n\n")
                    info_problem.write(dico[id_pgn][0]+  "/" + id_pgn + "/" + dico[id_pgn][1] + "\n")
            i+=1
        f.close()

def test_number_game_in_pgn(path_file_pgn):
    with open(path_file_pgn) as f:
        i=0
        while True:
            pgn=chess.pgn.read_game(f)
            if pgn is None:
                break
            i+=1
            
    print(i)
            

    
def create_pgn_problem_year(year, dico):
    
    prefix  = '/home/antoine/Bureau/projects/ParrotChess/Refactored/problem_data/lichess_db_standard_rated_' + year + '-'
    months = [ '0'+ str(i) for i in range(1,10)] + ['10', '11', '12']
    suffix = '.pgn'
    list_path = [(month, prefix + month + suffix) for month in months]
    list_input = [(elt[0], elt[1], dico) for elt in list_path]
    def aux(month, path_file_pgn, dico):
        pgn_problem = open("Refactored/problem_data/problem_" + year + '-' + month + ".pgn", "a")
        info_problem = open("Refactored/problem_data/info_problem_" + year + '-' + month + ".pgn", "a")
        i = 0
        m=0
        with open(path_file_pgn, 'r') as f:
            while True:
                if m*10000< i:
                    print(m*10000)
                    m+=1
                pgn = chess.pgn.read_game(f)
                if pgn is None:
                    print('done')
                    print(i)
                    break
                if pgn.next() is not None:
                    id_pgn = pgn.headers['Site'].split('/')[-1]
                    assert len(id_pgn) == 8, f'Invalid parse id {i}'
                    
                    if id_pgn in dico:
                        pgn_problem.write(str(pgn) + "\n\n")
                        info_problem.write(dico[id_pgn][0]+  "/" + id_pgn + "/" + dico[id_pgn][1] + "\n")
                i+=1
            f.close()
        pgn_problem.close()
        info_problem.close()
    processes = []
    for input in list_input:
        processes.append(Process(target=aux, args=input))
        
    for process in processes[:6]:
        print('start process')
        process.start()
    for process in processes[:6]:
        process.join()
    for process in processes[6:]:
        print('start process')
        process.start()
    for process in processes[6:]:
        process.join()
def merge_file(path_file_receiver, path_file_to_move):
    f_receiver = open(path_file_receiver, 'a')
    f_to_move = open(path_file_to_move, "r")
    f_receiver.write(f_to_move.read())

def old_main():
    dico = create_dict()
    L = ['/home/antoine/Bureau/projects/ParrotChess/Refactored/problem_data/lichess_db_standard_rated_2014-0' + str(i) + '.pgn' for i in range(1,10)]
    L.extend('/home/antoine/Bureau/projects/ParrotChess/Refactored/problem_data/lichess_db_standard_rated_2014-1' + str(i) + '.pgn' for i in range(0,3))
    for index,elt in enumerate(L):
        print('we are at month', index+1)
        keep_pgn_problem(elt, dico)  
        
if __name__ == '__main__':
    dico = create_dict()
    create_pgn_problem_year("2016", dico)
    print('end of 2016 parsing')
    create_pgn_problem_year("2017", dico)