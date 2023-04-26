from tkinter import *

class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.number = 0
        self.widgets = []
        self.geometry("780x650")
        self.grid()
        self.createWidgets()

    def createWidgets(self):
        self.cloneButton = Button ( self, text='Clone', command=self.clone)
        self.cloneButton.grid()

    def clone(self):
        widget = Label(self, text='label #%s' % self.number)
        widget.grid()
        self.widgets.append(widget)
        self.number += 1


if __name__ == "__main__":
    app = Application()
    app.master.title("Sample application")
    app.mainloop()