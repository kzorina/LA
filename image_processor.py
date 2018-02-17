import os
import numpy as np
from PIL import Image


def weightedAverage(pixel):
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]


def average(pixel):
    return (pixel[0] + pixel[1] + pixel[2])/3


def convert_to_grayscale(image):
    grey = np.zeros((image.shape[0], image.shape[1])) # init 2D numpy array
    # get row number
    for rownum in range(len(image)):
        for colnum in range(len(image[rownum])):
            grey[rownum][colnum] = weightedAverage(image[rownum][colnum])
    return grey


def fill_corners(image, standard_size):
    result = np.full(standard_size,255)
    offset = int((standard_size[1]-image.shape[1])/2)
    for rownum in range(len(image)):
        for colnum in range(len(image[rownum])):
            result[rownum][colnum+offset] = image[rownum][colnum]
    return result


def process_image(image, image_path, standard_size):

    base_height = standard_size[0]
    hpercent = (base_height / float(image.size[1]))
    wsize = int((float(image.size[0]) * float(hpercent)))
    image = image.resize((wsize, base_height), Image.ANTIALIAS)
    image = convert_to_grayscale(np.asarray(image))
    if wsize <= base_height:
        image = fill_corners(image, standard_size)
    else:
        print('Teach me for images where height is less then width')
    new_path = image_path.split('.')[0]+'.png'
    image_to_save = Image.fromarray(image.astype(np.uint8))
    image_to_save.save(new_path)
    os.remove(image_path)
    return image, new_path

