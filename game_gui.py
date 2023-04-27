import tkinter as tk
from tkinter.ttk import *
from tkinter import *
from PIL import Image, ImageTk

window = tk.Tk()
window.geometry('790x720')
window.configure(bg='white')
window.title('Connect 4')

button_frame = Frame(window, bg='white')
button_frame.pack()

canvas_frame=Frame(window, bg='white')
canvas_frame.pack()

col = tk.IntVar()

button0 = Button(button_frame, text="1", command = lambda: game.drop_piece(0))
button0.grid(row=0, column=0, padx=40, pady=10)
button1 = Button(button_frame, text="2", command = lambda: game.drop_piece(1))
button1.grid(row=0, column=1, padx=38, pady=10)
button2 = Button(button_frame, text="3", command = lambda: game.drop_piece(2))
button2.grid(row=0, column=2, padx=38, pady=10)
button3 = Button(button_frame, text="4", command = lambda: game.drop_piece(3))
button3.grid(row=0, column=3, padx=38, pady=10)
button4 = Button(button_frame, text="5", command = lambda: game.drop_piece(4))
button4.grid(row=0, column=4, padx=38, pady=10)
button5 = Button(button_frame, text="6", command = lambda: game.drop_piece(5))
button5.grid(row=0, column=5, padx=38, pady=10)
button6 = Button(button_frame, text="7", command = lambda: game.drop_piece(6))
button6.grid(row=0, column=6, padx=38, pady=10)

canvas = Canvas(canvas_frame, bg="white", width=790, height=700)
canvas.grid(row=1, column=0)

img_refs = []

# piecex = 123
# piecey = 95

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
            row = '| (' + ') ('.join(row) + ') |'
            print(row)

    def drop_piece(self, col):
        # drop a piece in the given column
        if col < 0 or col > 6:
            print('Invalid column, please give column 0-6')
        else:
            for row in range(5, -1, -1):
                if self.board[row][col] == 'e':
                    if self.turn == 'r':
                        self.playRed(row, col)
                        self.board[row][col] = 'r'
                        self.turn = 'y'
                    else:
                        self.playYellow(row, col)
                        self.board[row][col] = 'y'
                        self.turn = 'r'
                    break
        if self.check_win():
            if self.turn == 'y':
                winner = 'Red'
            else:
                winner = 'Yellow'
            self.forget()
            Label(window, text="Player " + winner + " wins!", font=("Arial", 30), bg="white").pack()
        elif self.check_tie():
            print('Tie!')
            self.forget()

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

    def playRed(self, row, col):
        load = Image.open('red.png')
        render = ImageTk.PhotoImage(load)

        x = 123 + 92*col
        y = 95 + 92*row

        canvas.create_image(x, y, image=render, tags=("pieces"))
        img_refs.append(render)
        canvas.tag_lower("pieces")

    def playYellow(self, row, col):
        load = Image.open('yellow.png')
        render = ImageTk.PhotoImage(load)

        x = 123 + 92*col
        y = 95 + 92*row

        canvas.create_image(x, y, image=render, tags=("pieces"))
        img_refs.append(render)
        canvas.tag_lower("pieces")

    def forget(self):
        for widgets in window.winfo_children():
            widgets.destroy()

board = Image.open('Connect4Board.png')
bgimg = ImageTk.PhotoImage(board)

photoimage = ImageTk.PhotoImage(file="Connect4Board.png")
canvas.create_image(400, 350, image=photoimage)

# Button(window, text="Red", command = showRed).pack()
# Button(window, text="Yellow", command = showYellow).pack()

game = Connect4()

window.mainloop()

# x for left = 123, y for top = 95
# increase x by 92 for each column
# increase y by 92 for each row