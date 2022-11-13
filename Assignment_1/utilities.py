from board import Board
# How many more disks are in the terminal state from the initial state
def u1(board, diskColor):
    maxValue = 64 - 1 # E.g. starts from 1 black and x white and ends up with 64 black and 0 white
    return (board.colors[diskColor] - board.initialColors[diskColor]) / maxValue

# The sum of possible moves until the max depth
def u2(board):
    return board.legalMovesSum

# red_value = 100, green_val = 16, blue_val = 25, orange_val = 1
# Attention! Maybe Max_value renders smaller-value evaluations too small, need to check if value returned is << 1.
def u3(board, diskColor):
    values = [16, 25, 1, 100]

    # Board Position Values
    posValues = [[values[0] for _ in range(8)] for k in range(8)]

    for i in range(1, 4):
        for pos in getPosValues()[i]:
            posValues[pos[0]][pos[1]] = values[i] * (1 if i != 2 else board.colors[0] + board.colors[1])

    maxValue = values[0] * 32 + values[1] * 16 + 64 * values[2] * 12 + values[3] * 4

    totalValue = 0
    for i in range(len(board.board)):
        for j in range(len(board.board[0])):
            if board.board[i][j] == diskColor:
                totalValue += posValues[i][j]
    
    return totalValue / maxValue

# the amount of legal moves the opponent has
# 1 - normalized value
def u4(board):
    return 1 / board.opponentLegalMoves

def getPosValues():
    redPositions = [(0, 0), (0, 7), (7, 0), (7, 7)]
    orangePositions = [(0, 1), (1, 0), (1, 1), (0, 6), (1, 7), (1, 6), (6, 0), (6, 1), (7, 1), (7, 6), (6, 6), (6, 7)]
    bluePositions = [(0, i) for i in range(2, 6)] + [(i, 0) for i in range(2, 6)] + [(7, i) for i in range(2, 6)] + [(i, 7) for i in range(2, 6)]

    return [None, bluePositions, orangePositions, redPositions]