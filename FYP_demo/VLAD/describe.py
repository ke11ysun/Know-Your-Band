import argparse
import glob
import cv2
import sys, os

import numpy as np 
import itertools
import pickle

def getDescriptors(path):
    descriptors=list()

    for imagePath in glob.glob(path+"/*.jpg"):
        print(imagePath)
        img = cv2.imread(imagePath)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(img,None)
        if np.any(des)!=None:
            descriptors.append(des)
            print(len(kp))
        
    descriptors = list(itertools.chain.from_iterable(descriptors))
    descriptors = np.asarray(descriptors)
    # print(descriptors.shape) (291008, 128)
    return descriptors


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required = True, help = "Path to where the computed descriptors will be stored")
ap.add_argument("-p", "--image_path", required = True)
args = vars(ap.parse_args())

output = args["output"]
path = args["image_path"]
# path = "/home/comp/e4252392/discogs_hot"

descriptors = getDescriptors(path)

file = output+".pickle"
with open(file, 'wb') as f:
	print("Dumping...")
	pickle.dump(descriptors, f)
print("Thank God written to file \m/")
