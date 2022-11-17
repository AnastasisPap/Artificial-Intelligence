from player import Player
from board import Board
from numpy import random
import random as rand


def tournament(weights):
    while len(weights) > 1:
        newWeights = []
        for j in range(1, len(weights), 2):
            if random.uniform() <= 0.5:
                newWeights.append(battle(weights[j-1], weights[j])[0])
            else:
                newWeights.append(battle(weights[j], weights[j-1])[0])
        weights = newWeights

    return weights[0]


def battle(blackWeights, whiteWeights):
    board = Board()
    checkers = [30, 30]
    turn = 1
    maxDepth = 2

    blackPlayer = Player(maxDepth, 1, blackWeights)
    whitePlayer = Player(maxDepth, 0, whiteWeights)
    players = [whitePlayer, blackPlayer]

    while checkers[0] + checkers[1] > 0:
        if board.hasLegalMove(turn):

            moveCoords = players[turn].miniMax(board)
            board.makeMove(moveCoords[0], moveCoords[1], turn)

            if checkers[turn] == 0:
                checkers[1 - turn] -= 1
                checkers[turn] += 1
                checkers[turn] -= 1

        if board.isTerminal(): break

        turn = 1 - turn

    if board.colors[0] > board.colors[1]:
        return whiteWeights, 0
    elif board.colors[0] < board.colors[1]:
        return blackWeights, 1
    else:
        if random.uniform() <= 0.5:
            return whiteWeights, 0
        else:
            return blackWeights, 1


# Using Modified Kraemer Algorithm to generate uniformly distributed pi s.t sum(pi) = 1,Note: every pi <> 0
def generateStartingWeights(numOfTuples, numOfWeights):
    startingWeightsSet = set([])
    startingWeights = []

    i = 0
    while i < numOfTuples:
        tuple = generateTuple(numOfWeights, 1)
        if str(tuple) not in startingWeightsSet:
            startingWeightsSet.add(str(tuple))
            startingWeights.append(tuple)
            i += 1
    
    return startingWeights


def generateTuple(numOfWeights, sm):
    M = 1000000
    t = rand.sample(range(M+1), numOfWeights - 1)
    t.append(0)
    t.append(M)
    t.sort()
    y = []
    for i in range(1, len(t)):
        y.append((t[i]-t[i-1])/M*sm)

    return tuple(y)


def determine(numOfTuples, numOfWeights):
    finale = []
    for i in range(numOfTuples):
        print(f"Tournament number {i+1}")
        finale.append(tournament(generateStartingWeights(numOfTuples, numOfWeights)))

    print(tournament(finale))