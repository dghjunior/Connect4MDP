# import dependencies

class Connect4:
    def __init__(self):
        self.new_board()
        self.turn = 'r'

    def new_board(self):
        # create a new board with 'e' for empty in every position
        self.board = [['e' for x in range(7)] for y in range(6)]
        self.turn = 'r'

    def print_board(self):
        # print the board
        for row in self.board:
            row = '| (' + ') ('.join(row) + ') |'
            print(row)

    def drop_piece(self, col):
        # drop a piece in the given column
        if col == None:
            return
        if col < 0 or col > 6:
            print('Invalid column, please give column 0-6')
        else:
            for row in range(5, -1, -1):
                if self.board[row][col] == 'e':
                    self.board[row][col] = self.turn
                    self.turn = 'y' if self.turn == 'r' else 'r'
                    break

    def check_win(self):
        # check if the game is won
        winner = 'r' if self.turn == 'y' else 'y'
        for row in range(6):
            for col in range(7):
                if self.board[row][col] != 'e':
                    if self.check_win_from(row, col):
                        return True, winner
        return False, 'e'
    
    def check_win_from(self, row, col):
        # check if the game is won from the given position
        # check horizontal
        ## check right
        if col < 4:
            if self.board[row][col] == self.board[row][col+1] == self.board[row][col+2] == self.board[row][col+3]:
                return True
        ## check left
        if col > 2:
            if self.board[row][col] == self.board[row][col-1] == self.board[row][col-2] == self.board[row][col-3]:
                return True
        # check vertical
        ## check up
        if row < 3:
            if self.board[row][col] == self.board[row+1][col] == self.board[row+2][col] == self.board[row+3][col]:
                return True
        ## check down
        if row > 2:
            if self.board[row][col] == self.board[row-1][col] == self.board[row-2][col] == self.board[row-3][col]:
                return True
        # check diagonal
        ## check up-right
        if row < 3 and col < 4:
            if self.board[row][col] == self.board[row+1][col+1] == self.board[row+2][col+2] == self.board[row+3][col+3]:
                return True
        ## check down-left
        if row > 2 and col > 2:
            if self.board[row][col] == self.board[row-1][col-1] == self.board[row-2][col-2] == self.board[row-3][col-3]:
                return True
        ## check up-left
        if row < 3 and col > 2:
            if self.board[row][col] == self.board[row+1][col-1] == self.board[row+2][col-2] == self.board[row+3][col-3]:
                return True
        ## check down-right
        if row > 2 and col < 4:
            if self.board[row][col] == self.board[row-1][col+1] == self.board[row-2][col+2] == self.board[row-3][col+3]:
                return True
        return False
    
    def check_tie(self):
        # check if the game is tied
        for row in range(6):
            for col in range(7):
                if self.board[row][col] == 'e':
                    return False
        return True
    
    def available_moves(self):
        moves = []
        for index in range(7):
            if 'e' in self.board[0][index]:
                moves.append(index)              
        return moves
    
    def play(self):
        # play the game
        while True:
            self.print_board()
            print(game.available_moves())
            if self.check_win()[0]:
                print('Player', 'r', 'wins!') if self.turn == 'y' else print('Player', 'y', 'wins!')
                break
            if self.check_tie():
                print('Tie!')
                break
            col = int(input('Player ' + self.turn + ', choose a column: '))
            self.drop_piece(col)

if __name__ == '__main__':
    game = Connect4()
    game.play()