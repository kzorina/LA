import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

STANDARD_SIZE = [243,320]
DATA_INPUT = 'train_process'
DATA_OUTPUT = 'train'


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


def fill_corners(image):
    result = np.full((STANDARD_SIZE[0], STANDARD_SIZE[1]),255)
    offset = int((STANDARD_SIZE[1]-image.shape[1])/2)
    for rownum in range(len(image)):
        for colnum in range(len(image[rownum])):
            result[rownum][colnum+offset] = image[rownum][colnum]
    return result


base_height = STANDARD_SIZE[0]
for file in os.listdir(DATA_INPUT):
    image_path = os.path.join(DATA_INPUT, file)
    image = Image.open(image_path)
    #plt.figure()
    #plt.imshow(image, cmap=plt.cm.gray)
    hpercent = (base_height / float(image.size[1]))
    wsize = int((float(image.size[0]) * float(hpercent)))
    image = image.resize((wsize, base_height), Image.ANTIALIAS)
    image = convert_to_grayscale(np.asarray(image))



    if (wsize <= base_height):
        image = fill_corners(image)
    else:
        print('Learn me for images where height is less then width')
    # image = resize(np.asarray(image),(wsize, baseheight), cval=0,mode='constant')
    # image = rescale(image,0.25,mode='reflect')

    # image = rescale(image,(61,80),mode='constant')
    # image_vector = image.flatten()
    # face_matrix.append(image_vector)
    plt.figure()
    plt.imshow(image, cmap=plt.cm.gray)
    #plt.show()
    image = Image.fromarray(image.astype(np.uint8))
    image.save(os.path.join(DATA_OUTPUT, file+'.png'))

