This directory contains the following :
- Mask_RCNN : the modified Mask RCNN library used. run "python3 setup.py install" from within this directory to install it before running any other code

- silique.py : the final code containing the final configuration used for training and detection. this file should stay in the same relative path to the library, new_dataset and the other packages used for it to function properly. details on how to run it are included in the comments at the top of the file

- MULTIVIEW : package containing the implementation of the Multi-transform and space optimization algorithms

- MODELVAL : package containing the implementation of the validation method

- NEW_DATASET : the restructured dataset

- V2-2_WEIGHTS.h5 : the final weights of the trained V2.2 model

- CROPER : contains the scripts that deal with image analysis. as well as the datasets generated using them (a copy of these datasets is found in the new_dataset directory)
	* crop_generator : generates the random crop images containing a minimum configurable amount of k siliques
	* space_optimizer_train.py : contains code for annotation based space optimization 
		(run using command "python3 space_optimizer_train.py path/to/image.PNG" or"python3 space_optimizer_train.py new_dataset/white/train/" )
	* space_optimizer_infer.py : contains the initial code implementing color based space optimization before it was integrated into the main code (run using "python3 space_optimizer_infer.py path/to/image/")


-LOGS : contains the redirected output of the validation and training processes. name suffixes are used to give overview of configuration used. e.g. no suffix means its training output, _mv means multi-transform (multi-view) was used, _so means space optimization was used. the final model validation output is named V2-2SOMV.log

-SAMPLE DETECTION : Contains detection examples. the images are chosen because they each present a different challenge to the detector

-EXPERIMENTS :  Contains the different configurations tested. each of these files need to be on the same directory as silique.py to run properly. uses the same commands.

