#image stitcher
from PIL import Image
from tkinter import filedialog
import os

def merge_images(file1, file2):
    image1 = Image.open(file1)
    image2 = Image.open(file2)

    (width1, height1) = image1.size
    (width2, height2) = image2.size

    result_width = max(width1, width2)
    result_height = height1 + height2

    result = Image.new('RGB', (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(0, height1))

    return result

im1 = filedialog.askopenfilename(initialdir="./", title="Select image file", filetypes=(("Image files (.jpg, .jpeg, .png)", "*.jpg *.jpeg *.png"), ("all files","*.*")))
im2 = filedialog.askopenfilename(initialdir="./", title="Select image file", filetypes=(("Image files (.jpg, .jpeg, .png)", "*.jpg *.jpeg *.png"), ("all files","*.*")))
stitch = merge_images(im1, im2)

if 'stitched_images' not in os.listdir('./'):
	os.mkdir('./stitched_images')
    
filepath = './stitched_images/' + os.path.split(im1)[1].split('.')[0] + '+' + os.path.split(im2)[1]
stitch.save(filepath)
print("saved to " + filepath)