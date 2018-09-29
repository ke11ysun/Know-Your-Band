import argparse
import glob
import cv2
import numpy as np 
from sklearn.cluster import KMeans
import pickle

def  kMeansDictionary(training, k):
    print('Start...')
    est = KMeans(n_clusters=k,init='k-means++',tol=0.0001,verbose=1,n_init=1).fit(training)
    print("Finish clustering.")
    return est


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--descriptorsPath", required = True, help = "Path to the file that contains the descriptors")
ap.add_argument("-o", "--output", required = True, help = "Path to where the computed visualDictionary will be stored")
args = vars(ap.parse_args())

path = args["descriptorsPath"]
k = 512
output=args["output"]

# print("estimating a visual dictionary of size: "+str(k)+ " for descriptors in path:"+path)
# print("Loading SIFT...")
# with open(path, 'rb') as f:
#     descriptors=pickle.load(f)
# print("SIFT loaded.")
# visualDictionary=kMeansDictionary(descriptors,k)

# file=output+".pickle"
# print("Dumping...")
# with open(file, 'wb') as f:
# 	pickle.dump(visualDictionary, f)
# print("Thank God written to file \m/")
# print("The visual dictionary is saved in "+file)





# k = 16
k=32 # for same dimension for CCA
print("estimating a visual dictionary of size: "+str(k)+ " for descriptors in path:"+path)
print("Loading CNN feature...")
with open(path, 'rb') as f:
    cnngt=pickle.load(f)
    raw_scenes = cnngt[1]
    print(len(raw_scenes))
    colf = [] # COLumn Feature
    for raw_scene in raw_scenes:
        raw_scene = raw_scene[0] # (1,2,x,2048)->(2,x,2048)
        print(raw_scene.shape)
        rows = raw_scene.shape[0]
        columns = raw_scene.shape[1]
        for i in range(rows):
            for j in range(columns):
                colf.append(raw_scene[i,j])
                # print(raw_scene[i,j].shape)
    print(len(colf)) # should be sum(rows*columns)
print("Column features loaded.")
visualDictionary=kMeansDictionary(colf,k)

file=output+".pickle"
print("Dumping...")
with open(file, 'wb') as f:
	pickle.dump(visualDictionary, f)
print("Thank God written to file \m/")
print("The visual dictionary is saved in "+file)





