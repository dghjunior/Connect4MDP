import tkinter as tk
from tkinter.ttk import *
from tkinter import *
from PIL import Image, ImageTk

# Piece Location Info:
## x for left = 123, y for top = 95
## increase x by 92 for each column
## increase y by 92 for each row

img_refs = []

class Connect4:
    def __init__(self):
        self.window = tk.Tk()
        self.turn = 'r'
        self.new_board()

    def new_board(self):
        # create a new board with 'e' for empty in every position
        self.board = [['e' for x in range(7)] for y in range(6)]
        self.turn = 'r'
        self.window.destroy()

        self.window = tk.Tk()
        self.window.geometry('790x720')
        self.window.configure(bg='white')
        self.window.title('Connect 4')

        button_frame = Frame(self.window, bg='white')
        canvas_frame = Frame(self.window, bg='white')
        self.canvas = Canvas(canvas_frame, bg="white", width=790, height=700)

        button_frame.pack()
        canvas_frame.pack()

        button0 = Button(button_frame, text="1", command = lambda: self.drop_piece(0))
        button0.grid(row=0, column=0, padx=40, pady=10)
        button1 = Button(button_frame, text="2", command = lambda: self.drop_piece(1))
        button1.grid(row=0, column=1, padx=38, pady=10)
        button2 = Button(button_frame, text="3", command = lambda: self.drop_piece(2))
        button2.grid(row=0, column=2, padx=38, pady=10)
        button3 = Button(button_frame, text="4", command = lambda: self.drop_piece(3))
        button3.grid(row=0, column=3, padx=38, pady=10)
        button4 = Button(button_frame, text="5", command = lambda: self.drop_piece(4))
        button4.grid(row=0, column=4, padx=38, pady=10)
        button5 = Button(button_frame, text="6", command = lambda: self.drop_piece(5))
        button5.grid(row=0, column=5, padx=38, pady=10)
        button6 = Button(button_frame, text="7", command = lambda: self.drop_piece(6))
        button6.grid(row=0, column=6, padx=38, pady=10)

        self.canvas.grid(row=1, column=0)

        board = Image.open('Connect4Board.png')
        bgimg = ImageTk.PhotoImage(board)

        # photoimage = ImageTk.PhotoImage(file="Connect4Board.png")
        self.canvas.create_image(400, 350, image=bgimg)
        self.canvas.image = bgimg

        self.window.mainloop()

    def print_board(self):
        # print the board
        for row in self.board:
            row = '| (' + ') ('.join(row) + ') |'
            print(row)

    def end_screen(self, winner):
        self.forget()
        if winner == 'tie':
            Label(self.window, text="It's a tie!", font=("Arial", 30), bg="white").pack()
        else:
            Label(self.window, text="Player " + winner + " wins!", font=("Arial", 30), bg="white").pack()
        Button(self.window, text="Play again", command=self.new_board).pack()

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
            self.end_screen(winner)
        elif self.check_tie():
            self.end_screen('tie')

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

        self.canvas.create_image(x, y, image=render, tags=("pieces"))
        img_refs.append(render)
        self.canvas.tag_lower("pieces")

    def playYellow(self, row, col):
        load = Image.open('yellow.png')
        render = ImageTk.PhotoImage(load)

        x = 123 + 92*col
        y = 95 + 92*row

        self.canvas.create_image(x, y, image=render, tags=("pieces"))
        img_refs.append(render)
        self.canvas.tag_lower("pieces")

    def forget(self):
        for widgets in self.window.winfo_children():
            widgets.destroy()

game = Connect4()