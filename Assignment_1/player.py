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

    def maxValue(self, board, depth, a, b):
        if board.isTerminal() or depth == self.maxDepth:
            return self.h1(board)

        currMax = float('-inf')
        maxCoords = (-1, -1)

        legalMoves = board.getLegalMoves(1)
        if len(legalMoves) == 0:
            legalMoves.append([-1, -1])

        for child in legalMoves:
            newBoard = board.getCopy()
            newBoard.makeMove(child[0], child[1], 1)
            currValue = self.minValue(newBoard, depth + 1, a, b)
            if currValue >= b: 
                self.optCoords = maxCoords
                return currValue
            if currValue >= currMax:
                if currValue == currMax and random.uniform() <= 0.5:
                    continue
                maxCoords = child
                currMax = currValue
            a = max(a, currValue)
        
        self.optCoords = maxCoords
        return currMax

    def minValue(self, board, depth, a, b):
        if board.isTerminal() or depth == self.maxDepth:
            return self.h1(board)

        currMin = float('inf')
        minCoords = (-1, -1)

        legalMoves = board.getLegalMoves(0)
        if len(legalMoves) == 0:
            legalMoves.append([-1, -1])

        for child in legalMoves:
            newBoard = board.getCopy()
            newBoard.makeMove(child[0], child[1], 0)
            currValue = self.maxValue(newBoard, depth + 1, a, b)
            if currValue <= a: 
                self.optCoords = minCoords 
                return currValue
            if currValue <= currMin:
                if currValue == currMin and random.uniform() <= 0.5:
                    continue
                minCoords = child
                currMin = currValue
            b = min(b, currValue)

        self.optCoords = minCoords
        return currMin
    
    def h1(self, board):
        return randint(0, 10)