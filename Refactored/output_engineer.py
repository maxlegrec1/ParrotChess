import numpy as np
import matplotlib.pyplot as plt

mask = np.zeros(shape = (15,15))

# Create kings moves
kings_move = [(i,j) for i in range(-1,2) for j in range(-1, 2) if i!=0 and j!=0]

# Create knights moves
knights_move = [(i,j) for i in range(-2,3) for j in range(-2,3) if abs(i) + abs(j) == 3]

# Create bishops moves
bishops_move = [(i,i*j) for i in range(-7,8) for j in [-1,1] if i != 0]

# Create rooks moves
rooks_move = [(i,j) for i in range(-7,8) for j in range(-7,8) if (i == 0 or j == 0) and (i != 0 or j!= 0)]
legal_moves = kings_move + knights_move + bishops_move + rooks_move

for pos in legal_moves:
    mask[7+pos[0],7+pos[1]] = 1
# the mask have been verified, you can imshow the mask to verify
plt.imshow(mask,cmap='gray')
plt.show()
extended_length= 7+8+7
board = np.zeros(shape =(8,8))
map_of_moves=[]
for i in range(8):
    for j in range(8):
        
        # All legal moves from (i,j) in the board
        masked_board = mask[7-i:8+(7-i),7-j:8+(7-j)] + board
        legal_moves = np.where(masked_board == 1)
        # Add legal moves from (i,j) to the list of legal moves
        for k in range(len(legal_moves[0])):
            map_of_moves.append((i + 8*j, legal_moves[0][k]+8* legal_moves[1][k]))
        # plt.imshow(masked_board,cmap='gray')
        # plt.show()
print(len(map_of_moves))
print(map_of_moves)
# We need to add promotions
# For each column, we add the piece to add (after moving up or down for the given pawn)
# 65 is rook, 66 is knight, 67 is bishop, 68 is queen
for i in range(8):
    for j in range(4):
        map_of_moves.append((i,65+j))
print(len(map_of_moves))