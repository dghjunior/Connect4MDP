# import dependencies
import numpy as np
import matplotlib
import tkinter as tk
from tkinter import *
import Connect4
from PIL import ImageTk, Image

class game_gui:
    def __init__(self):
        window = tk.Tk()
        window.title("Connect 4")
        window.geometry("780x650")
        window.maxsize(780,650)

        frame = Frame(window)
        frame.pack()

        canvas = Canvas(frame, bg="white", width=780, height=650)
        canvas.pack()
        
        # Create a photoimage object of the image in the path
        background = ImageTk.PhotoImage(file="Connect4Board.png")
        canvas.create_image(390, 325, image=background)

        red_piece = PhotoImage(file='red.png')
        # x for left = 113, y for top = 73
        # increase x by 92 for each column
        # increase y by 92 for each row
        canvas.create_image(297, 533, image=red_piece)
        yellow_piece = PhotoImage(file='yellow.png')
        canvas.create_image(113, 533, image=yellow_piece)

        window.mainloop()

        
gui = game_gui()