from board import Board

def main():
    board = Board()
    board.printBoard()

    player_color = ['White', 'Black']
    checkers = [30, 30]
    turn = 1
    
    while not board.full() and checkers[0] + checkers[1] > 0:
        
        if board.hasLegalMove(turn):
            print(f"{player_color[turn]} plays!")

            while True:
                
                # Input validation required
                r = int(input("Input row: "))
                c = int(input("Input col: "))

                if not board.makeMove(r, c, turn):
                    continue

                if checkers[turn] ==  0:
                    checkers[1-turn] -= 1
                    checkers[turn] += 1
                checkers[turn] -= 1
                break
        else:
            print(f"{player_color[turn]} looses his turn since there isn't any legal move for him.")
        
        board.printBoard()
        if board.winnerExists(): break

        turn = not turn

    if board.colors[0] > board.colors[1]:
        print('White player wins!')
    else:
        print('Black player wins!')            
              
main()