# import dependencies

class Connect4:
    def __init__(self):
        self.new_board()
        self.turn = 'r'

    def new_board(self):
        # create a new board with 'e' for empty in every position
        self.board = [['e' for x in range(7)] for y in range(6)]

    def print_board(self):
        # print the board
        for row in self.board:
            print(row)

    def drop_piece(self, col):
        # drop a piece in the given column
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
        for row in range(6):
            for col in range(7):
                if self.board[row][col] != 'e':
                    if self.check_win_from(row, col):
                        return True
        return False
    
    def check_win_from(self, row, col):
        # check if the game is won from the given position
        # check horizontal
        if col <= 3:
            if self.board[row][col] == self.board[row][col+1] == self.board[row][col+2] == self.board[row][col+3]:
                return True
        # check vertical
        if row <= 2:
            if self.board[row][col] == self.board[row+1][col] == self.board[row+2][col] == self.board[row+3][col]:
                return True
        # check diagonal down
        if row <= 2 and col <= 3:
            if self.board[row][col] == self.board[row+1][col+1] == self.board[row+2][col+2] == self.board[row+3][col+3]:
                return True
        # check diagonal up
        if row >= 3 and col <= 3:
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
    
    def play(self):
        # play the game
        while True:
            self.print_board()
            if self.check_win():
                print('Player', self.turn, 'wins!')
                break
            if self.check_tie():
                print('Tie!')
                break
            col = int(input('Player ' + self.turn + ', choose a column: '))
            self.drop_piece(col)

if __name__ == '__main__':
    game = Connect4()
    game.play()