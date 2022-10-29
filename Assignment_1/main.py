from player import Player
from board import Board

def main():
    board = Board()
    board.printBoard()

    player_color = ['White', 'Black']
    checkers = [30, 30]
    turn = 1
    
    player = Player(2)
    while checkers[0] + checkers[1] > 0:
        
        if board.hasLegalMove(turn):
            print(f"{player_color[turn]} plays!")

            if turn == 0:
                while True:
                    
                    # Input validation required
                    r = int(input("Input row: "))
                    c = int(input("Input col: "))

                    if not board.makeMove(r, c, turn):
                        continue

                    
                    break
            else:
                moveCoords = player.miniMax(board)
                print(moveCoords)
                board.makeMove(moveCoords[0], moveCoords[1], 1)
                
            if checkers[turn] ==  0:
                checkers[1-turn] -= 1
                checkers[turn] += 1
                checkers[turn] -= 1
        else:
            print(f"{player_color[turn]} looses his turn since there isn't any legal move for him.")
        
        board.printBoard()
        if board.isTerminal(): break

        turn = 1 - turn

    if board.colors[0] > board.colors[1]:
        print('White player wins!')
    else:
        print('Black player wins!')            
              
main()