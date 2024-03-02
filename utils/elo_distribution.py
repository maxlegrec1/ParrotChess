import chess.pgn
import chess
import matplotlib.pyplot as plt
from tqdm import tqdm
with open("human.pgn") as f:
    elo_arrays = []
    game_lengths = []
    for i in tqdm(range(1000000)):
        pgn = chess.pgn.read_game(f)
        try:
            white_elo = int(pgn.headers["WhiteElo"])
        except:
            white_elo = 1500
        try:
            black_elo = int(pgn.headers["BlackElo"])
        except:
            black_elo = 1500
        moves = [move for move in pgn.mainline_moves()]
        game_lengths.append(len(moves))
        elo_arrays.append(white_elo)
        elo_arrays.append(black_elo)

#plot historgram of elo distribution and game lengths

plt.hist(elo_arrays,bins=100)
plt.show()
plt.hist(game_lengths,bins=100)
plt.show()
