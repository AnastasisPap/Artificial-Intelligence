class PossibleMoves:
    def __init__(self):
        self.hasMoves = False
        self.columnRange = (-1, -1)
        self.rowRange = (-1, -1)
        self.leftGoingUp = [(-1, -1)] * 2
        self.leftGoingDown = [(-1, -1)] * 2
    
    def addHorizontal(self, newColumnRange):
        if newColumnRange != -1:
            self.hasMoves = True
            self.columnRange = newColumnRange

    def addVertical(self, newRowRange):
        if newRowRange != -1:
            self.hasMoves = True
            self.rowRange = newRowRange

    def addDiagonals(self, diagonals):
        if diagonals[0] != -1 or diagonals[1] != -1:
            self.hasMoves = True 

        if diagonals[0] != -1: self.leftGoingUp = diagonals[0] 
        if diagonals[1] != -1: self.leftGoingDown = diagonals[1] 

class Board:
    def __init__(self):
        # -1 = Empty spot, 0 = white, 1 = black
        self.board = [[-1 for _ in range(8)] for k in range(8)]
        self.board[3][3] = 0
        self.board[3][4] = 1
        self.board[4][3] = 1
        self.board[4][4] = 0

        self.colors = [2, 2]


    def full(self):
        return self.colors[0] + self.colors[1] == 64


    def printBoard(self):
        dict = {-1 : "   ", 1 : ' ● ', 0 : ' ○ '}
        t = "―" * 30

        for r in self.board:
            print(t)
            s = ''
            for c in r: 
                s += '|' + dict[c]
            print(s + '|')
        print(t)
        print(self.colors)

    def hasLegalMove(self, player):
        for r in range(len(self.board)):
            for c in range(len(self.board[0])):
                if self.board[r][c] == -1:
                    possibleMoves = self.findMoves(r, c, player)
                    if possibleMoves.hasMoves: return True


        return False
        

    def findMoves(self, row, col, diskColor):
        possibleMoves = PossibleMoves()
        if row >= 8 or col >= 8 or row < 0 or col < 0 or self.board[row][col] != -1: return possibleMoves

        possibleMoves.addHorizontal(self.outflankHorizontally(row, col, diskColor))
        possibleMoves.addVertical(self.outflankVertically(row, col, diskColor))
        possibleMoves.addDiagonals(self.outflankDiagonally(row, col, diskColor))

        return possibleMoves
    

    def makeMove(self, row, col, diskColor):
        possibleMoves = self.findMoves(row, col, diskColor)
        if not possibleMoves.hasMoves:
            return False

        columnRange = possibleMoves.columnRange
        rowRange = possibleMoves.rowRange
        leftGoingUp = possibleMoves.leftGoingUp
        leftGoingDown = possibleMoves.leftGoingDown
        
        self.board[row][col] = diskColor
        self.colors[diskColor] += 1

        for currCol in range(columnRange[0] + 1, columnRange[1]):
            self.board[row][currCol] = diskColor
            self.colors[diskColor] += 1
            self.colors[1-diskColor] -= 1
        
        for currRow in range(rowRange[0] + 1, rowRange[1]):
            self.board[currRow][col] = diskColor
            self.colors[diskColor] += 1
            self.colors[1-diskColor] -= 1
            
        i = 0
        while leftGoingUp[0][1] + i < leftGoingUp[1][1]:
            self.board[leftGoingUp[0][0] - i][leftGoingUp[0][1] + i] = diskColor
            if i > 0: 
                self.colors[diskColor] += 1
                self.colors[1-diskColor] -= 1
            i += 1

        i = 0
        while leftGoingDown[0][1] + i < leftGoingDown[1][1]:
            self.board[leftGoingDown[0][0] + i][leftGoingDown[0][1] + i] = diskColor
            if i > 0: 
                self.colors[diskColor] += 1
                self.colors[1-diskColor] -= 1
            i += 1

        return True


    def outflankHorizontally(self, row, col, diskColor):
        _, leftCol = self.move(row, col, 0, -1, diskColor)
        _, rightCol = self.move(row, col, 0, 1, diskColor)

        if leftCol == rightCol == -1: return -1

        if leftCol == -1: leftCol = col
        
        if rightCol == -1: rightCol = col
        
        return (leftCol, rightCol)


    def outflankVertically(self, row, col, diskColor):
        upRow, _ = self.move(row, col, -1, 0, diskColor)
        downRow, _ = self.move(row, col, 1, 0, diskColor)

        if upRow == downRow == -1: return -1

        if upRow == -1: upRow = row

        if downRow == -1: downRow = row

        return (upRow, downRow)


    def outflankDiagonally(self, row, col, diskColor):
        leftUp = self.move(row, col, -1, -1, diskColor)
        leftDown = self.move(row, col, 1, -1, diskColor)
        rightUp = self.move(row, col, -1, 1, diskColor)
        rightDown = self.move(row, col, 1, 1, diskColor)
 
        leftGoingUp = self.makeDiagonal(row, col, leftDown, rightUp)
        leftGoingDown = self.makeDiagonal(row, col, leftUp, rightDown)
        return leftGoingUp, leftGoingDown


    def makeDiagonal(self, row, col, leftCorner, rightCorner):
        if leftCorner == [-1, -1] and rightCorner == [-1, -1]: return -1

        if leftCorner != [-1, -1]:
            if rightCorner != [-1, -1]:
                return [leftCorner, rightCorner]
            return [leftCorner, (row, col)]
        else:
            return [(row, col), rightCorner]


    def move(self, startRow, startCol, stepRow, stepCol, diskColor):
        currCol = startCol + stepCol
        currRow = startRow + stepRow

        if currCol < 0 or currRow < 0: return [-1, -1]

        while 0 <= currCol < 8 and 0 <= currRow < 8:
            if self.board[currRow][currCol] == diskColor:
                break

            if self.board[currRow][currCol] == -1: return [-1, -1]
            currCol += stepCol
            currRow += stepRow
        
        if currCol == startCol + stepCol and currRow == startRow + stepRow or (currCol == 8 or currRow == 8):
            return [-1, -1]

        return [currRow, currCol]


    def winnerExists(self):
        return self.colors[0] == 0 or self.colors[1] == 0


    def testBoard(self):
        self.board[1][1] = 1
        self.board[2][2] = 1
        self.board[2][3] = 0
        self.board[2][4] = 1
        self.board[2][5] = 1
        self.board[1][5] = 1
        self.board[3][3] = 0
        self.board[3][4] = 1
        self.board[3][5] = 1
        self.board[4][1] = 0
        self.board[4][2] = 0
        self.board[4][3] = 0
        self.board[4][4] = 1
        self.board[4][5] = 0
        self.board[4][6] = 0
        self.board[4][7] = 0
        self.board[5][1] = 0
        self.board[5][1] = 0
        self.board[5][5] = 1
        self.board[5][3] = 0
        self.board[6][3] = 0
        self.board[6][6] = 1
        self.board[6][7] = 0
        self.board[7][7] = 1