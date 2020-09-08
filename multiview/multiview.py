

import skimage
import os
import sys
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

ROOT_DIR = os.path.abspath("../Mask_RCNN")
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize as vis
from math import cos, sin




class InferenceConfig(Config) :
    NAME = "silique"
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + Silique
    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_MAX_INSTANCES = 200
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MAX_DIM = 1344
    RPN_NMS_THRESHOLD = 0.9



def nms(bboxes, masks, scores, thresh) : # non-maximum suppression
    assert boxes.shape[0] > 0
    final_bboxes = []
    final_masks = []
    overlaps = utils.compute_overlaps(bboxes, bboxes)
    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

                
def translatecoord(h, w, bbox, side) :
    # coordinate translation function : translate bounding box coordinates between rotated image spaces
    y1, x1, y2, x2 = bbox
    if side == "right" :
        ydiff = abs(x1 - x2) # difference of y in the final image is the the difference of xs in the original
        ny1 = w - x1 - ydiff
        nx1 = y1 
        ny2 = w - x2 + ydiff
        nx2 = y2 
    elif side == "left" :
        xdiff = abs(y1 - y2) # difference of x in the final image is the the difference of ys in the original
        ny1 = x1 
        nx1 = h - y1 - xdiff
        ny2 = x2 
        nx2 = h - y2 + xdiff
    elif side == "upsidedown" :
        ydiff = abs(y1 - y2)
        ny1 = h - y1 - ydiff
        nx1 = x1
        ny2 = h - y2 + ydiff
        nx2 = x2 
 
    return [ny1, nx1, ny2, nx2]


def multiview_detect(model, im, standard=False, verbose=True) :
    # Implements Multi-Transform detection (see multi-transform detection chapter in report)
    od = model.detect([im])[0]  
    if standard :
        # if standard more activated then return the first, normal , detection
        return od
    # run 3 more detections on rotated and flipped image
    r90_im = skimage.transform.rotate(im, 90, resize=True, preserve_range=True)
    r90 = model.detect([r90_im])[0] 
    rn90_im = skimage.transform.rotate(im, -90, resize=True, preserve_range=True)
    rn90 = model.detect([rn90_im])[0] 
    ud_im = im[::-1]
    ud = model.detect([ud_im])[0] 

    # get translated bounding boxes
    r90_bbx = np.array([translatecoord(len(r90_im), len(r90_im[0]), b, "left") for b in r90['rois']] )
    rn90_bbx = np.array([translatecoord(len(rn90_im), len(rn90_im[0]), b, "right") for b in rn90['rois']] )
    ud_bbx = np.array([translatecoord(len(ud_im), len(ud_im[0]), b, "upsidedown") for b in ud['rois']] )


    # rotate masks (todo: move this code to function similar to translatecoord)
    rnm = rn90['masks'][:, :, 0]
    rn90_mask = np.zeros([len(rnm[0]), len(rnm), rn90['masks'].shape[-1]], dtype=np.uint8)
    for i in range(rn90['masks'].shape[-1] ) :
             tmp = rn90['masks'][:,:,i]
             tmp = tmp.transpose()
             tmp = tmp[::-1]
             rn90_mask[:, :, i] = tmp

    rm = r90['masks'][:, :, 0]
    r90_mask = np.zeros([len(rm[0]), len(rm), r90['masks'].shape[-1]], dtype=np.uint8)
    for i in range(r90['masks'].shape[-1] ) :
             tmp = r90['masks'][:,:,i]
             tmp = tmp[::-1]
             tmp = tmp.transpose()
             r90_mask[:, :, i] = tmp

    for i in range(ud['masks'].shape[-1]) :
            tmp = ud['masks'][:, : , i] 
            ud['masks'][:, :, i] = tmp[::-1]

    # make sure the masks are the same shape to ensure a successful overlay
    if od["masks"].shape != r90_mask.shape or od["masks"].shape != rn90_mask.shape :
       r90_mask.resize((len(od['masks']), len(od['masks'][0]), r90['masks'].shape[-1]))
       rn90_mask.resize((len(od['masks']), len(od['masks'][0]), rn90['masks'].shape[-1]))
    
    # merge masks and bounding box vectors in preparation for the nonm-maximum suppression process
    all_masks = np.concatenate((od['masks'], ud['masks'], r90_mask, rn90_mask), axis=2)
    all_bboxes = np.concatenate((od['rois'], ud_bbx, r90_bbx, rn90_bbx))
    all_scores = np.concatenate(( od['scores'], ud['scores'], r90['scores'], rn90['scores']))
    all_class_ids = np.concatenate(( od['class_ids'], ud['class_ids'], r90['class_ids'], rn90['class_ids']))

    final = utils.non_max_suppression(all_bboxes, all_scores, 0.55)
    # final contains the indices of the boxes to be kept, these are used to extract them as well as the masks, scores and other metadata corresponding to each
    final_masks = np.take(all_masks, final, axis=2) 
    final_bboxes= np.take(all_bboxes, final, axis=0)
    final_scores = np.take(all_scores, final)
    final_class_ids = np.take(all_class_ids, final)
    if verbose :
        print("standard detection instances : " + str(len(od['rois'])) )
        print("90 deg detection instances : " + str(len(r90_bbx)) )
        print("-90 deg detection instances : " + str(len(rn90_bbx)) )
        print("upside down detection instances : " + str(len(ud_bbx)) )
        print("post nms detect : " + str(len(final_bboxes)))
    #vis.display_instances(im, final_bboxes, final_masks, final_class_ids, ['', ''] , save=True, name="multiview_detect_{}_{}".format(sys.argv[2].split('/')[-1].split('.')[0], sys.argv[3]), savedir=".")
    return {
            'rois' : final_bboxes,
            'scores' : final_scores,
            'masks' : final_masks,
            'class_ids' : final_class_ids
        }

###### Space Optimization ########

def get_clr_dist(s, d):
    # get euclidean distance between colours s and d in 3D space
    dist = math.sqrt(math.pow(d[0] - s[0], 2) + math.pow(d[1] - s[1], 2) + math.pow(d[2] - s[2], 2))
    return dist

def get_min_max(im):
    # find edges based on colour distance and return box bounds
    min_x, max_x, min_y, max_y = 0, 0, 0, 0
    threshold = 25 # threshold distance between colours used to determine edges

    print("Scanning Y column...")
    for r in range(1, int(len(im)/2)) : 
        for c in range(1, len(im[0])) :
            if min_y and max_y :
                break
            if get_clr_dist(im[r, c], [141, 143, 87]) < threshold and not min_y:
                min_y = r 
                print("min y found at {}.".format(min_y))

            if get_clr_dist(im[len(im) - 1 - r, c], [141, 143, 87]) < threshold and not max_y :
                max_y = len(im) - 1 - r
                print("max y found at {}.".format(max_y))

    print("Scanning X column...")
    for c in range(1, int(len(im[0])/2)) :
        for r in range(min_y, max_y) : # only search in already established y limits to make search faster
            if min_x and max_x :
                break
            if get_clr_dist(im[r, c], [141, 143, 87]) < threshold and not min_x :
                min_x = c
                print("min x found at {}.".format(min_x))
            if get_clr_dist(im[r, len(im[0])- 1 - c], [141, 143, 87]) < threshold and not max_x :
                max_x = len(im[0]) - 1 - c
                print("max x found at {}.".format(max_x))
       
    return min_x, max_x, min_y, max_y                     



def getCrop(im, bounds):
    # get image portion contain within bounds
    h, w = im.shape[:2]
    im_dtype = im.dtype
    new_im = im[bounds[2]:bounds[3], bounds[0]:bounds[1]]
    return new_im.astype(im_dtype)

def space_optimize(im, pad=0) :
    min_x, max_x, min_y, max_y = get_min_max(im)
    min_x -= pad
    max_x += pad
    min_y -= pad
    max_y += pad
    bounds = [min_x , max_x, min_y, max_y]
    so_im = getCrop(im, bounds)
    return so_im, bounds, im.shape


if __name__ == "__main__" :
    model = modellib.MaskRCNN(mode="inference", config=InferenceConfig(), model_dir="./")
    model.load_weights(sys.argv[1], by_name=True)


    im = skimage.io.imread(sys.argv[2])
    so_im = space_optimize(im)
    bboxes = multiview_detect(model, im)
