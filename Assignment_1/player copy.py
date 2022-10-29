from random import randint
from numpy import random 
from board import Board

class Player: 
    def __init__(self, maxDepth):
        self.maxDepth = maxDepth
        self.optCoords = (-1, -1)
    
    def miniMax(self, board):
       self.maxValue(board, 0) 
       return self.optCoords

    def maxValue(self, board, depth):
        if board.isTerminal() or depth == self.maxDepth:
            return self.h1(board)

        currMax = float('-inf')
        maxCoords = (-1, -1)
        for child in board.getChildren():
            newBoard = board.getCopy()
            currValue = self.minValue(newBoard.makeMove(child[0], child[1], 1), depth + 1)
            if currValue >= currMax:
                if currValue == currMax and random.uniform() <= 0.5:
                    continue
                maxCoords = child
        
        self.optCoords = maxCoords
        return currMax

    def minValue(self, board, depth):
        if board.isTerminal() or depth == self.maxDepth:
            return self.h1(board)

        currMin = float('inf')
        minCoords = (-1, -1)
        for child in board.getChildren():
            newBoard = board.getCopy()
            currValue = self.maxValue(newBoard.makeMove(child[0], child[1], 0), depth + 1)
            if currValue <= currMin:
                if currValue == currMin and random.uniform() <= 0.5:
                    continue
                minCoords = child

        self.optCoords = minCoords
        return currMin
    
    def h1(self, board):
        return randint(0, 10)