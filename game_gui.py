import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk

window = tk.Tk()
window.geometry('790x800')

frame=Frame(window)
frame.pack()

canvas = Canvas(frame, bg="white", width=790, height=700)
canvas.pack()

img_refs = []

piecex = 123
piecey = 95

def showRed():
    global piecex
    global piecey
    load = Image.open('red.png')
    render = ImageTk.PhotoImage(load)

    canvas.create_image(piecex, piecey, image=render, tags=("pieces"))
    img_refs.append(render)
    canvas.tag_lower("pieces")

    piecex = piecex + 0
    piecey = piecey + 92

def showYellow():
    global piecex
    global piecey
    load = Image.open('yellow.png')
    render = ImageTk.PhotoImage(load)

    canvas.create_image(piecex, piecey, image=render, tags=("pieces"))
    img_refs.append(render)
    canvas.tag_lower("pieces")

    piecex = piecex + 92
    piecey = piecey + 92

board = Image.open('Connect4Board.png')
bgimg = ImageTk.PhotoImage(board)

photoimage = ImageTk.PhotoImage(file="Connect4Board.png")
canvas.create_image(400, 350, image=photoimage)

# label1 = Label(image=bgimg)
# label1.image = bgimg
# label1.place(x=0, y=0)

Button(window, text="Red", command = showRed).pack()
Button(window, text="Yellow", command = showYellow).pack()

window.mainloop()

# x for left = 113, y for top = 73
# increase x by 92 for each column
# increase y by 92 for each row