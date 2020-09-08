#!/usr/bin/python3

import skimage
import sys
import random
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mplp
import math

from PIL import Image, ImageDraw

def getRandomCrop(min_dim_x, min_dim_y, im):
        print("min dimensions: ({},{})".format(min_dim_y, min_dim_x))
        h, w = im.shape[:2]
        im_dtype = im.dtype
        print("image type: {}".format(im_dtype))
        print("image dimensions: ({},{})".format(h, w))
        y = random.randint(0, (h - min_dim_y))
        x = random.randint(0, (w - min_dim_x))
        crop = (y, x, min_dim_y, min_dim_x)
        new_im = im[y:(y + min_dim_y), x:(x + min_dim_x)]
        plt.imshow(new_im)
        plt.savefig("crop.png")
        return new_im.astype(im_dtype), crop


def getClrDist(s, d):
    dist = math.sqrt(math.pow(d[0] - s[0], 2) + math.pow(d[1] - s[1], 2) + math.pow(d[2] - s[2], 2))
    return dist 
    

def plot_masks(im, xp, yp):
    pol_coord = ()
    c = np.array(xp)
    r = np.array(yp)
    rr, cc = skimage.draw.polygon(r, c)
    im[rr, cc] = 1
    return im

def contains_polygons(cim, crop, annots):

    counter = 0
    print(crop)
    crop = (crop[0], crop[1], crop[0] + crop[2], crop[1] + crop[3])
    with open("/home/cpt-n3m0/MMP/Silique-Detector/new_dataset/annots/" + annots + ".csv") as annot :
        data = csv.reader(annot)
        for row in data :
                if(row[5][0] != "{") :
                        continue
                jsn = json.loads(row[5])
                x_pos = jsn['all_points_x']
                y_pos = jsn['all_points_y']

                crop_coord = mplp.Path(np.array([
                    [crop[0], crop[1]],
                    [crop[1], crop[2]],
                    [crop[2], crop[3]],
                    [crop[3], crop[0]]
                ]))
                contained = True
                for i in range(len(x_pos)) :
                    #contained = crop_coord.contains_point(( x_pos[i], y_pos[i]))
                    #if not contained:
                    #    break
                    if y_pos[i] - crop[0] < 0 or  x_pos[i] - crop[1] < 0 or y_pos[i] > crop[2] or x_pos[i] > crop[3] :
                        contained = False
                        break
                nx_pos, ny_pos = [], []   
                if contained :
                    print ("----- row {} ---- ".format(str(counter + 1)))
                    print(x_pos)
                    print(y_pos)
                    counter += 1
                    # adapt mask pos
                    for i in range(len(x_pos)) :
                        nx_pos.append(x_pos[i] - crop[1])
                        ny_pos.append(y_pos[i] - crop[0])
                    print("------ updated row {}".format(str(counter)))
                    print(nx_pos)
                    print(ny_pos)
                    cim = plot_masks(cim, nx_pos, ny_pos)
                    
                    contained = False
        print ("{} masks found.".format(str(counter)))               
        return counter


def threshold_and_average(im):
        h, w = im.shape[:2]
        non_bg = 0
        for y in range(h):
           for x in range(w) :
                if getClrDist(im[y][x], [255, 255, 255]) < 130 :
                    im[y][x] = [255, 226, 10]
                else :
                    non_bg += 1

        return im, non_bg
                    

print(sys.argv)
im = skimage.io.imread(sys.argv[1])
npoly = 0
while npoly < 2 :
    cim, crop = getRandomCrop(int(sys.argv[2]), int(sys.argv[3]), im)
    npoly = contains_polygons(cim, crop, sys.argv[1].split(".")[0].split("/")[-1])

thcim, sc = threshold_and_average(cim)
plt.imshow(thcim)
plt.savefig("thresholded_crop_.png")
print("silique pixel count : " + str(sc))


