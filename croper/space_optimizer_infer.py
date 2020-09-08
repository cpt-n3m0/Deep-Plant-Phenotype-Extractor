#!/usr/bin/python3

import csv
import os
import sys
import json
import math
import numpy as np
import skimage
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.affinity import scale, rotate
from PIL import Image, ImageDraw



def get_clr_dist(s, d):
    dist = math.sqrt(math.pow(d[0] - s[0], 2) + math.pow(d[1] - s[1], 2) + math.pow(d[2] - s[2], 2))
    return dist

def get_min_max_old(im):
    min_x, max_x, min_y, max_y = 0, 0, 0, 0
    for r in range(len(im)) : 
        for c in range(len(im[0])) :
            if get_clr_dist(im[r, c], [141, 143, 87]) > 130 :
                min_x = c if c < min_x else min_x
                max_x = c if c > max_x else max_x
                min_y = r if r < min_y else min_y
                max_y = r if r > max_y else max_y

def get_min_max(im):
    min_x, max_x, min_y, max_y = 0, 0, 0, 0
    threshold = 25
    print("START SIZE : {}x{}".format(len(im), len(im[0])))
    print("Scanning Y column...")
    for r in range(int(len(im)/2)) : 
        for c in range(len(im[0])) :
            if min_y and max_y :
                break
            if get_clr_dist(im[r, c], [141, 143, 87]) < threshold and not min_y:
                min_y = r 
                print("min y found at {}.".format(min_y))
                print("dist : " + str(get_clr_dist(im[r, c], [141, 143, 87])))
                print("color : " + str(im[r, c]))

            if get_clr_dist(im[len(im) - 1 - r, c], [141, 143, 87]) < threshold and not max_y :
                max_y = len(im) - 1 - r
                print("max y found at {}.".format(max_y))
                print("dist : " + str(get_clr_dist(im[len(im) - 1 - r, c], [141, 143, 87])))
                print("color : " + str(im[len(im) - 1 - r, c]))

    print("Scanning X column...")
    for c in range(int(len(im[0])/2)) :
        for r in range(len(im)) :
            if min_x and max_x :
                break
            if get_clr_dist(im[r, c], [141, 143, 87]) < threshold and not min_x :
                min_x = c
                print("min x found at {}.".format(min_x))
                print("dist : " + str(get_clr_dist(im[r, c], [141, 143, 87])))
                print("color : " + str(im[r, c]))
            if get_clr_dist(im[r, len(im[0])- 1 - c], [141, 143, 87]) < threshold and not max_x :
                max_x = len(im[0]) - 1 - c
                print("max x found at {}.".format(max_x))
                print("dist : " + str(get_clr_dist(im[r, len(im[0]) - 1 - c], [141, 143, 87])))
                print("color : " + str(im[r, len(im[0]) - 1 - c]))
       
    return min_x, max_x, min_y, max_y                     



def getCrop(im, bounds):
    h, w = im.shape[:2]
    im_dtype = im.dtype
    new_im = im[bounds[2]:bounds[3], bounds[0]:bounds[1]]
    print("cropped.")
    return new_im.astype(im_dtype)

def plot_bounding_box(im, min_x, max_x, min_y, max_y):
        draw = ImageDraw.Draw(im)
        draw.line((min_x, min_y, min_x, max_y, max_x, max_y, max_x, min_y, min_x, min_y), fill=128, width=2)
        im.show()

im = skimage.io.imread( sys.argv[1])
img = Image.open(sys.argv[1])

pad = 0
min_x, max_x, min_y, max_y = get_min_max(im)
min_x -= pad
max_x += pad
min_y -= pad
max_y += pad

plot_bounding_box(img, min_x, max_x, min_y, max_y)
so_im = getCrop(im, [min_x , max_x, min_y, max_y])
print("new size: {}x{}".format(len(so_im), len(so_im[0])))
print("optimization: {}%".format(((len(im) * len(im[0])) - (len(so_im) * len(so_im[0]))) / (len(im) * len(im[0]))) )
plt.imsave("processed.png", so_im)

