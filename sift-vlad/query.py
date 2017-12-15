import itertools
import argparse
import glob
import cv2
import pickle
import numpy as np
import os

def VLAD(X,visualDictionary):
    predictedLabels = visualDictionary.predict(X)
    centers = visualDictionary.cluster_centers_
    labels=visualDictionary.labels_
    k=visualDictionary.n_clusters
   
    m,d = X.shape
    V=np.zeros([k,d])
    for i in range(k):
        if np.sum(predictedLabels==i)>0:
            V[i]=np.sum(X[predictedLabels==i,:]-centers[i],axis=0)
    
    V = V.flatten()
    V = np.sign(V)*np.sqrt(np.abs(V))
    V = V/np.sqrt(np.dot(V,V))
    return V


def query(image, k, visualDictionary,tree):
    img = cv2.imread(image)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, descriptor = sift.detectAndCompute(img,None)

    v=VLAD(descriptor,visualDictionary)
    dist, ind = tree.query(v.reshape(1,-1), k)
    return dist, ind


ap = argparse.ArgumentParser()
ap.add_argument("-r", "--retrieve", required = True,
	help = "number of images to retrieve")
args = vars(ap.parse_args())


img_path = "/home/comp/e4252392/VLAD-master/test1118"
k=int(args["retrieve"])
pathVD = "/home/comp/e4252392/VLAD-master/discogs_hot_vd.pickle"
treeIndex = "/home/comp/e4252392/VLAD-master/discogs_hot_tree.pickle"

with open(treeIndex, 'rb') as f:
    indexStructure=pickle.load(f)
with open(pathVD, 'rb') as f:
    visualDictionary=pickle.load(f)
imageID=indexStructure[0]
tree = indexStructure[1]
pathImageData = indexStructure[2]
print(pathImageData)

for idx, img_name in enumerate(sorted(os.listdir(img_path))):
    if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
        continue
    print(img_name)
    filepath = os.path.join(img_path, img_name)
    dist,ind = query(filepath, k, visualDictionary,tree)
    print(dist)
    print(ind)
    ind = list(itertools.chain.from_iterable(ind))

    print(filepath)
    for i in ind:
        print(imageID[i])
