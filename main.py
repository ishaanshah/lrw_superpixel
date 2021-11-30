import tkinter as tk
import cv2
import math
import sys
import numpy as np
import time

from PIL import Image, ImageTk
from tkinter import filedialog

from src.lrw import generate_seeds, energy_opt
from src.utils import im2double, seg2bmap

new_imgtk = None

tk.Tk().withdraw()
image_name = filedialog.askopenfilename()

image = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_RGB2BGR)

height, width = image.shape[:2]
lrw_target_width = 200
lrw_target_height = math.floor(200 / width * height)
smaller_image = cv2.resize(image, (lrw_target_width, lrw_target_height),
                           interpolation=cv2.INTER_AREA)

# A root window for displaying objects
root = tk.Toplevel()
root.title("Lazy Random Walk for Superpixel Segmentation")


im = Image.fromarray(image)

# Calculate target image size
# Get screen dimensions (-100 for title bars and stuff)
screen_width = root.winfo_screenwidth() - 100
screen_height = root.winfo_screenheight() - 100

# Get target dimensions (-100 for space needed for controls)
padding = 300
target_width = screen_width - padding
target_height = screen_height
ratio = min(target_width / width, target_height / height)
im = im.resize((math.floor(width*ratio),
                math.floor(height*ratio)), Image.ANTIALIAS)

# Convert the Image object into a TkPhoto object
imgtk = ImageTk.PhotoImage(image=im)

canvas = tk.Canvas(root, width=screen_width, height=screen_height)
canvas.pack()

image_container = canvas.create_image(0, 0, image=imgtk, anchor=tk.NW)
controls_start = target_width + padding // 2
canvas.create_text(controls_start, 40, anchor=tk.CENTER,
                   font=("Arial", 20), text="Parameters")


def create_entry(name, y, init_value):
    seed_count_label = tk.Label(root, text=name)
    seed_count_label.config(font=('Arial', 10))
    canvas.create_window(controls_start, y, window=seed_count_label)
    seed_count = tk.StringVar()
    seed_count.set(init_value)
    seed_count_entry = tk.Entry(canvas, textvariable=seed_count)
    seed_count_entry.pack()
    canvas.create_window(controls_start, y+25, window=seed_count_entry)
    return seed_count


seed_count = create_entry("Seed Count", 100, 200)
beta = create_entry("Beta", 150, 30)
alpha = create_entry("Lazy Parameter", 200, 0.99)
threshold = create_entry("Threshold", 250, 1.35)
max_iters = create_entry("Max Iterations", 300, 10)


def entrypoint():
    gray_img = cv2.cvtColor(smaller_image.astype("uint8"), cv2.COLOR_RGB2GRAY)
    seeds = generate_seeds(int(seed_count.get()), im2double(gray_img / 255))

    label_img, _, _ = energy_opt(smaller_image,
                                 seeds,
                                 float(alpha.get()),
                                 int(seed_count.get()),
                                 int(max_iters.get()),
                                 float(beta.get()),
                                 float(threshold.get()))

    bmap = seg2bmap(label_img, smaller_image.shape[1], smaller_image.shape[0])
    idx = np.nonzero(bmap > 0)

    bmap_on_img = smaller_image.copy()
    if len(bmap_on_img.shape) == 3:
        bmap_on_img[idx[0], idx[1], 0] = 255
        bmap_on_img[idx[0], idx[1], 1] = 0
        bmap_on_img[idx[0], idx[1], 2] = 0
    else:
        bmap_on_img[idx[0], idx[1]] = 0

    x, y = bmap_on_img.shape[:2]
    new_ratio = min(target_width / y,
                    target_height / x)
    new_image = Image.fromarray(bmap_on_img)
    new_image = new_image.resize((math.floor(y*new_ratio),
                                  math.floor(x*new_ratio)), Image.ANTIALIAS)
    new_image.save("result.png")
    global new_imgtk
    new_imgtk = tk.PhotoImage(file="result.png")
    global canvas
    global image_container
    canvas.itemconfig(image_container, image=new_imgtk)


submit_button = tk.Button(canvas, command=entrypoint,
                          text="Generate", anchor=tk.CENTER)
canvas.create_window(controls_start, 400, window=submit_button)

root.mainloop()
