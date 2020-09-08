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
import os

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

def contains_polygons(cim, crop, annots, generate_annot=False):
    #check if cim contains annotated siliques
    # in case generate_annot flag is true then we will need a path to store the annots in
    counter = 0
    regions = []
    print(crop)
    crop_bounds = (crop[0], crop[1], crop[0] + crop[2], crop[1] + crop[3]) # (x, y, h, w) becomes (y, x, y + h, x + w)
    with open("/home/cpt-n3m0/MMP/Silique-Detector/new_dataset/annots/" + annots + ".csv") as annot :
        data = csv.reader(annot)
        for row in data :
                if(row[5][0] != "{") :
                        continue
                jsn = json.loads(row[5])
                x_pos = jsn['all_points_x']
                y_pos = jsn['all_points_y']

                contained = True
                for i in range(len(x_pos)) :
                    if y_pos[i] - crop_bounds[0] < 0 or  x_pos[i] - crop_bounds[1] < 0 or y_pos[i] > crop_bounds[2] or x_pos[i] > crop_bounds[3] :
                        contained = False
                        break
                if contained :
                    counter += 1
                    if generate_annot :
                        # adapt mask pos
                        nx_pos, ny_pos = [], []   
                        for i in range(len(x_pos)) :
                            nx_pos.append(x_pos[i] - crop_bounds[1])
                            ny_pos.append(y_pos[i] - crop_bounds[0])
                        tmp =  {
                                "name" : "polygon", 
                                "all_points_x" : nx_pos, 
                                "all_points_y" : ny_pos 
                                } 
                        regions.append(tmp)
                    
    print ("{} masks found.".format(str(counter))) 
           
    return counter, regions

def write_annot(path, regions, imName):
    # generte annotation data for the new crop
    # file format :
    # filename, file_size, file_attributes, region_count, region_id, region_shape_attributes, region_attributes
    with open(path + "/" + imName + ".csv", 'w' ) as new_annot :
        annot_writer = csv.writer(new_annot, delimiter=',')
        annot_writer.writerow(["filename", "file_size", "file_attributes", "region_count", "region_id", "region_shape_attributes", "region_attributes"])
     
        file_size = "fileSize" 
        file_attributes = {
                        "Width": str(sys.argv[2]),
                        "Height": str(sys.argv[3]),
                        "Type":"PNG"
                    }
        region_count = len(regions)
        region_id = 0 
        region_attributes = {"Part" : "Silique"}
        for region in regions :
            annot_writer.writerow([imName, file_size, json.dumps(file_attributes), region_count, region_id, json.dumps(region), json.dumps(region_attributes)])
            region_id += 1
     
def threshold_and_average(im):
        # performs background subtraction 
        h, w = im.shape[:2]
        non_bg = 0
        for y in range(h):
           for x in range(w) :
                if getClrDist(im[y][x], [255, 255, 255]) < 130 : # allows for high distance threshold to include shadows as well in the subtraction
                    #im[y][x] = [255, 226, 10]
                    im[y][x] = [255, 255, 255]
                    
                else :
                    non_bg += 1 # counts number of non-background pixels (silique or stem pixels)

        return im, non_bg
                    


for img in os.listdir(sys.argv[1]):
    for i in range(20) :
        if "{}_c{}.png".format(img.split(".")[0], str(i)) in os.listdir(".") :
            print("skipped : " + "{}_c{}.png".format(img.split(".")[0], str(i)))
            continue
        im = skimage.io.imread(sys.argv[1] + "/" + img)
        npoly = 0
        while npoly < 2 :
            cim, crop = getRandomCrop(int(sys.argv[2]), int(sys.argv[3]), im)
            npoly, regions = contains_polygons(cim, crop, img.split(".")[0], generate_annot=True )
        thcim, sc = threshold_and_average(cim)
        plt.imshow(thcim)
        print("saving image ...{}_c{}.png".format(img.split(".")[0], str(i))) 
        plt.imsave("{}_c{}.png".format(img.split(".")[0], str(i)), cim)
        print("saving annotation ...{}_c{}".format(img.split(".")[0], str(i)))
        write_annot("./", regions, "{}_c{}".format(img.split(".")[0], str(i)) )
        print("silique pixel count : " + str(sc))

 



