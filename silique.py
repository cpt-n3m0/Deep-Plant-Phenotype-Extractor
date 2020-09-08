"""
Mask R-CNN
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 silique.py train --dataset=/path/to/balloon/dataset --weights=coco --modelname=modelname

    # Resume training a model that you had trained earlier
    python3 silique.py train --dataset=/path/to/balloon/dataset --weights=last --modelname=modelname

    # Train a new model starting from ImageNet weights
    python3 silique.py train --dataset=/path/to/balloon/dataset --weights=imagenet --modelname=modelname

    # Train a new model starting from other weights
    python3 silique.py train --dataset=/path/to/balloon/dataset --weights=/path/to/weights/file.h5 --modelname=modelname

    # Detect Siliques
    python3 silique.py detect --weights=/path/to/weights/file.h5 --image=<URL or path to file> --modelname=modelname
  
    # Validate model (e.g. ... validate --dataset=new_dataset --weights=V2-2_weigths.h5 --modelname=V2-2)
    python3 silique.py validate --dataset=/path/to/dataset/dir --weights=/path/to/weights/file.h5 --modelname=modelname


"""

import os
import sys
import json
import re
import datetime
import numpy as np
import skimage.draw
import csv
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import imgaug
import time
from multiview import multiview
from modelval import validate as val
from shapely.geometry import Point
from shapely.affinity import scale, rotate

# Root directory of the project
ROOT_DIR = os.path.abspath("./Mask_RCNN")


used_imgs = []

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
from mrcnn.visualize import display_images, display_top_masks
from random import shuffle
from imgaug import augmenters as augs
from glob import glob
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################
class SiliqueConfig(Config):
    NAME = "silique"
    
    # We use a GPU with 12GB memory, which can fit two images.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + Silique

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 50% confidence (minimum since we have 2 classes)
    DETECTION_MIN_CONFIDENCE = 0.5

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MAX_DIM = 1280
    
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)

class InferenceConfig(SiliqueConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MAX_INSTANCES = 200
    IMAGE_MAX_DIM = 1344
    RPN_NMS_THRESHOLD = 0.9



############################################################
#  Dataset
############################################################
def fixSingleLineAnnot(xp, yp, imsize):
     # code of ellipse generation is written by user ewcz as answer in this thread https://stackoverflow.com/questions/45158360/how-to-create-ellipse-shape-geometry
    newxp, newyp = [], []
    A = Point(xp[0], yp[0])
    B = Point(xp[1], yp[1])
    R = 4
    d = A.distance(B)
    S = Point(A.x + d/2, A.y)
    alpha = math.atan2(B.y - A.y, B.x - A.x)
    C = S.buffer(d/2)
    C = scale(C, 1, R/(d/2))
    C = rotate(C, alpha, origin = A, use_radians = True)
    counter = 0
    for x,y in C.exterior.coords:
        # %5 to lower the size of the coordinates needed to create the ellipse to the fifth of the original
        # the equalities are to prevent us from creating coordinates outside the bounds of the pictures
        if counter % 5 == 0 and x < int(imsize['Width']) and y < int(imsize['Height']):
            newxp.append(int(x))
            newyp.append(int(y))
        counter += 1
    return newxp, newyp 


# The next three functions handle loading the annotation data from the dataset in the following format :
        #{ 'filename': '*_w.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   }
def extract_regions(annot_file, annotObj) :
    # open the annotation file and extract the polygon coordinates for each region 
    # store the results in annotObj
    with open(annot_file) as annot :
        data = csv.reader(annot)

        tmp_shape = {}
        tmp_size = {}
        counter = 0
        for row in data :
            if(row[5][0] != "{") : # skip header cell
                    continue
            if counter == 0: # Do it once after the headers
                tmp_size = json.loads(row[2]) # Load file_attributes column
            
            jsn = json.loads(row[5])
            x_pos = jsn['all_points_x']
            y_pos = jsn['all_points_y']
            if len(x_pos) == 3 :
                x_pos, y_pos = fixSingleLineAnnot(x_pos, y_pos, tmp_size)
                jsn['all_points_x'] = x_pos
                jsn['all_points_y'] = y_pos
            tmp_shape[str(counter)] = { 'shape_attributes' : jsn}
            counter += 1
            
        annotObj['regions'] = tmp_shape
        annotObj['file_attributes'] = tmp_size
        return annotObj


def extract_annots(dataset_dir, bckgrnd, subset) :
    annots = {}
    annot_lst = []
    counter =  0 
    for i in range(len(bckgrnd)) :
        # use the bckgrn and subset options to build the path to the dataset 
        wdir = dataset_dir + '/' + bckgrnd[i] + '/' + subset + '/'
        for img in os.listdir(wdir):
            
            counter += 1
            annots = {"filename" : wdir + img}
            annot_path = dataset_dir + '/annots/' + img.split('.')[0] + '.csv'
            annots = extract_regions(annot_path, annots)
            annot_lst.append(annots)
    print("Loaded " + str(counter) + " images")
    return annot_lst

       
class SiliqueDataset(utils.Dataset):

    def load_silique(self, dataset_dir, subset, bckgrnd):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("silique", 1, "silique")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        for c in bckgrnd :
            assert c in ["white", "black", "reduced", "reduced2", "cropped"] # list of existing datasets, when dataset is added, its name should be also added here

        annotations = extract_annots(dataset_dir, bckgrnd, subset)
        counter = 0
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stored in the
            # shape_attributes 
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 

            image_path = a['filename']
            # sanity check: we shouldn't use the same image twice. 
            if not image_path in used_imgs :
                used_imgs.append(image_path)
            else:
                continue

            try :
                height, width = int(a["file_attributes"]["Height"]), int(a["file_attributes"]["Width"])
            except KeyError:
                print("FILE CONTAINING FAILED INSTANCE: " + str(a["filename"]))
                print(str(a["file_attributes"]))
                sys.exit(-1)

            self.add_image(
                "silique",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)
            counter += 1

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a silique dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "silique":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            try :
                mask[rr, cc, i] = 1
            except IndexError:
                print("Index error :")
                print("Image Path: {}".format(image_info['path']))
                print("Image size: {}x{}".format(image_info['height'], image_info['width']))
                print("Polygon x: {}, y: {}".format(p['all_points_x'], p['all_points_y']))

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "silique":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, modelname=None):
    """Train the model."""
    assert modelname # sanity check
    # Training dataset.
    dataset_train = SiliqueDataset()
    dataset_train.load_silique(args.dataset, "train", ["white", "reduced"])
    dataset_train.prepare()
    # Validation dataset
    dataset_val = SiliqueDataset()
    dataset_val.load_silique(args.dataset, "val", ["white"])
    dataset_val.prepare()

    # augmentation config
    augmentation = augs.Sometimes(0.7, augs.SomeOf((1, 3),  [ 
                                            augs.Flipud(0.5), 
                                            augs.Fliplr(0.5),
                                            augs.GaussianBlur(sigma=(0.0, 5.0)),
                                            augs.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
                                            augs.Affine(rotate=(-90, 90))
                                            ]
                                            , random_order=True))
    
    print("Training the heads")
    start = time.time()
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='heads',
                augmentation=augmentation)
    end = time.time()
    print("Head training lasted : " + str(end - start))

    print("Fine tuning whole network")
    start = time.time()
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='all',
                augmentation=augmentation)
    end = time.time()
    print("Network tuning lasted : " + str(end - start))
    
    # rename log dir to a more recognisable name
    md = "Mask_RCNN/logs/{}/".format(modelname)
    os.rename(model.log_dir, md)


def examin_data(model, image_path=None, option=None):
    # visualization function. 4 basic options provided:
    # activation visualization
    # pre and post mold mask visualization
    # augmentations visualization
    #
    # change as necessary
    
    assert image_path or option 

    if image_path :
            
        print("Running on {}".format(args.image))
        
        image = skimage.io.imread(args.image)
        # run selected graphs and save outputs for each
        activations = model.run_graph([image], [
        ("input_image",        tf.identity(model.keras_model.get_layer("input_image").output)),
        ("res2c_out",          model.keras_model.get_layer("res2c_out").output),
        ("res3c_out",          model.keras_model.get_layer("res3c_out").output),
        ("res4w_out",          model.keras_model.get_layer("res4w_out").output),  # for resnet100
        ("rpn_bbox",           model.keras_model.get_layer("rpn_bbox").output),
        ("roi",                model.keras_model.get_layer("ROI").output),
        ])
        _ = plt.imshow(modellib.unmold_image(activations["input_image"][0], SiliqueConfig()))
        display_images(np.transpose(activations["res3c_out"][0,:,:,:5], [2, 0, 1]), cols=5, save=True, name="activations_aa_"+ args.image.split(".")[0], savedir = "Mask_RCNN/output")
    elif option :
        #prepare dataset and augmentation config
        dataset = SiliqueDataset()
        dataset.load_silique("new_dataset/", "val", [ "white"])
        dataset.prepare()
        image_ids = np.random.choice(dataset.image_ids, 10)
        augmentation = augs.Sometimes(0.7, augs.SomeOf((1, 3),  [
                                            augs.Flipud(0.5),
                                            augs.Flipud(0.5),
                                            augs.GaussianBlur(sigma=(0.0, 5.0)),
                                            augs.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
                                            augs.Affine(rotate=(-90, 90))
                                            ]
                                            , random_order=True))
 
    if option == "premold_masks":
        # view images and masks before mold
       for image_id in image_ids :
            print("extracting {}".format(image_id))
            image = dataset.load_image(image_id)
            mask, class_ids = dataset.load_mask(image_id)
            display_top_masks(image, mask, class_ids, dataset.class_names, limit=1, sve=True, nme="white_masks_{}".format(image_id), svedir="Mask_RCNN/output")
    elif option == "postmold_masks":
        print("Generating molded images...")
        for image_id in image_ids :
            print("extracting {}".format(image_id))
            #load image after it has been processed by the model
            image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(dataset, SiliqueConfig(), image_id, use_mini_mask=False, augmentation=augmentation)
            display_top_masks(image, mask, class_ids, dataset.class_names, limit=1, sve=True, nme="white_masks_{}".format(image_id), svedir="Mask_RCNN/output")




def detect_silique(model, image_path , show=True, mv=True):
    # image detector, model ran in inference mode with InferenceConfig 
    assert image_path 
    assert args.modelname
    # Read image
    print("Running on {}".format(image_path))
    image = skimage.io.imread(image_path)
    image = multiview.space_optimize(image)
    # model only handles 3 channel RGB images with shape (h, w, 3). remove additional channels (alpha channel) if necessary
    if image.shape[-1] > 3 :    
       image = image[:, :, :3] 
    r = multiview.multiview_detect(model, image, standard= not mv) # run multi-transform detection (see report for details)
    
    # prepare detection image storage directory. if doesn't exist create it.
    image_name = image_path.split('/')[-1].split('.')[0]
    output_dir = "Mask_RCNN/output/" + args.modelname
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], ['background', 'silique'], save=True, name="detection_{}_{}".format(image_name, args.modelname), savedir=output_dir, display=show)

    if not args.dataset : 
        args.dataset = "new_dataset/"
    return r



def validate(model):
    # run validation on model    
    dataset = SiliqueDataset()
    ds =  ["reduced2"] # dataset to test the model on
    dataset.load_silique(args.dataset, "val", ds)
    dataset.prepare()
    val.validate_model(model, dataset, save=True, modelname=args.modelname)



############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect silique.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect' or 'visualize'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/silique/dataset/",
                        help='Directory of the silique dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to detect siliques on')
    parser.add_argument('--vis', required=False,
                        metavar="visualization option",
                        help='What you would like to visualize')
    parser.add_argument('--modelname', required=False,
                        metavar="Model name",
                        help='What name would you like to give this model (Affects directory and output names)')

    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.modelname, "Please provide the --modelname"
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.image, "Provide --image to detect siliques"
    elif args.command == "visualize":
        assert args.image or args.vis, "Provide --image or --vis option to visualize"
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = SiliqueConfig()
    else:
       config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    elif args.weights.lower() == "lastinmodel" :
        # get latest weight file in logs directory
        md = "Mask_RCNN/logs/{}/".format(args.modelname)
        lof = glob(md + "/*")
        lof = [f for f in lof if f.split(".")[-1] == "h5"]
        weights_path = max(lof, key=os.path.getctime)

    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.modelname)
    elif args.command == "detect":
        detect_silique(model, image_path=args.image)
    elif args.command == "validate":
        validate(model)
    elif args.command == "visualize":
        if args.image :
            examin_data(model, image_path=args.image)
        else :
            examin_data(model, option=args.vis)

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
