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

base_path = "/home/cpt-n3m0/MMP/Silique-Detector/new_dataset/"

def get_min_max(xp, yp):
        
        if len(xp) == 3 :
                # code of ellipse generation is written by user ewcz as answer in this thread https://stackoverflow.com/questions/45158360/how-to-create-ellipse-shape-geometry
                newxp, newyp = [], []
                A = Point(xp[0], yp[0])
                B = Point(xp[1], yp[1])
                R = 3
                d = A.distance(B)
                S = Point(A.x + d/2, A.y)
                alpha = math.atan2(B.y - A.y, B.x - A.x)
                C = S.buffer(d/2)
                C = scale(C, 1, R/(d/2))
                C = rotate(C, alpha, origin = A, use_radians = True)
                counter = 0
                for x,y in C.exterior.coords:
                    if counter % 5 == 0 :
                        newxp.append(int(x))
                        newyp.append(int(y))
                    counter += 1
                xp = newxp
                yp = newyp
        return sorted(xp)[0], sorted(xp)[-1], sorted(yp)[0],  sorted(yp)[-1]


def translate_annot(annot, bounds):
    # bounds: [min_x, max_x, min_y, max_y]
    regions = []
    for row in annot :
        if(row[5][0] != "{") :
                continue
        jsn = json.loads(row[5])
        x_pos = jsn['all_points_x']
        y_pos = jsn['all_points_y']
        
        nx_pos =  [x - bounds[0] for x in x_pos ]
        ny_pos =  [y - bounds[2] for y in y_pos ]
        tmp =  {
                    "name" : "polygon",
                    "all_points_x" : nx_pos,
                    "all_points_y" : ny_pos
                }
        regions.append(tmp)

    return regions

def write_annot(regions, imName, width, height) :

    with open("so_ds/" + imName + ".csv", 'w' ) as new_annot :
            annot_writer = csv.writer(new_annot, delimiter=',')
            annot_writer.writerow(["filename", "file_size", "file_attributes", "region_count", "region_id", "region_shape_attributes", "region_attributes"])

            file_size = "fileSize"
            file_attributes = {
                            "Width": str(width),
                            "Height": str(height),
                            "Type":"PNG"
                        }
            region_count = len(regions)
            region_id = 0
            region_attributes = {"Part" : "Silique"}
            for region in regions :
                annot_writer.writerow([imName, file_size, json.dumps(file_attributes), region_count, region_id, json.dumps(region), json.dumps(region_attributes)])
                region_id += 1

def getCrop(im, bounds):
    h, w = im.shape[:2]
    im_dtype = im.dtype
    print("image type: {}".format(im_dtype))
    print("image dimensions: ({},{})".format(h, w))
    new_im = im[bounds[2]:bounds[3], bounds[0]:bounds[1]]
    return new_im.astype(im_dtype)

def plot_bounding_box(im, min_x, max_x, min_y, max_y):
        draw = ImageDraw.Draw(im)
        draw.line((min_x, min_y, min_x, max_y, max_x, max_y, max_x, min_y, min_x, min_y), fill=128, width=2)
        im.show()

def getbounds(img_name) :
    with open("../new_dataset/annots/" + img_name + ".csv") as annot :
            data = csv.reader(annot)
            max_x, min_x, max_y, min_y = 0, -1 , 0, -1
            pad = 20

            for row in data :
                if(row[5][0] != "{") :
                        continue
                jsn = json.loads(row[5])
                x_pos = jsn['all_points_x']
                y_pos = jsn['all_points_y']
                mx, x, my, y = get_min_max(x_pos, y_pos)
                max_x = x if x > max_x else max_x 
                max_y = y if y > max_y else max_y 
                min_x = mx if mx < min_x or min_x < 0 else min_x
                min_y = my if my < min_y or min_y < 0 else min_y
                
            min_x -= pad
            max_x += pad
            min_y -= pad
            max_y += pad
    return min_x, max_x, min_y, max_y


assert sys.argv[1], "Need dataset path"
if sys.argv[1].split('/')[-1].split(".")[-1] == "png" :
    img_name =  sys.argv[1].split('/')[-1].split(".")[0]
    print("image_name :" + img_name)
    imd = Image.open(sys.argv[1])
    min_x, max_x, min_y, max_y = getbounds(img_name) 
    plot_bounding_box(imd, min_x, max_x, min_y, max_y)
    sys.exit(0)

for img in os.listdir(sys.argv[1]) :
    if im_specific and sys.argv[1].split('/')[-1] != img :
            continue
    im = skimage.io.imread( sys.argv[1] + "/" + img)
    img_name = img.split(".")[0]
    min_x, max_x, min_y, max_y = getbounds(img_name) 
    annot.seek(0)
    so_im = getCrop(im, [min_x , max_x, min_y, max_y ])
    regions = translate_annot(data, [min_x, max_x, min_y, max_y]) 
    write_annot(regions, "{}_so".format(img_name), max_x - min_x, max_y - min_y)
    print("saved {}_so.csv".format(img_name))
    plt.imsave("so_ds/{}_so.png".format(img_name), so_im)
    print("saved {}_so.png".format(img_name))

