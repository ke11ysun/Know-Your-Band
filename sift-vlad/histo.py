import cv2
import numpy as np
import argparse
import pickle
import glob
import itertools

path = "/home/comp/e4252392/discogs_hot"

histos = list()
imageIDs = list()
for imagePath in glob.glob(path+"/*.jpg"):
    print(imagePath)
    imageIDs.append(imagePath.split("/")[-1])
    img=cv2.imread(imagePath)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    histos.append(hist)
print(imageIDs[0])
#histos = list(itertools.chain.from_iterable(histos))
#histos = np.asarray(histos)

file = "discogs_hot_histos.pickle"
with open(file, 'wb') as f:
	pickle.dump([imageIDs, histos, path], f)
print("Histograms are saved in "+file)




























