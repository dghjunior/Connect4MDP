# import dependencies
import numpy as np
import matplotlib
import tkinter as tk
import Connect4
from PIL import ImageTk, Image

class game_gui:
    def __init__(self):
        window = tk.Tk()
        
        # Create a photoimage object of the image in the path
        image1 = Image.open("Connect+4+Background.jpg")
        test = ImageTk.PhotoImage(image1)

        label1 = tk.Label(image=test)
        label1.image = test
        label1.pack()

        window.mainloop()
        
gui = game_gui()