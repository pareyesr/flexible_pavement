import tkinter as tk
from src import flexible_pavement_gui

if __name__ == "__main__":
    root = tk.Tk()
    app = flexible_pavement_gui.App(root)
    root.mainloop()