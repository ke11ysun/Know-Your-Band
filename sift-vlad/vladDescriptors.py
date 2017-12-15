import argparse
import glob
import cv2
import numpy as np 
import pickle
from sklearn.decomposition import PCA

def getVLADDescriptors(path,visualDictionary):
    #load cnn features
    with open("/home/comp/e4252392/cnntrees/discogs_hot_cnngt_pre.pickle", 'rb') as f:
        pkl = pickle.load(f)
        objfs = pkl[1]
        scenefs = pkl[2]
        img_names = pkl[0]
    new_objfs = []
    new_scenefs = []
    for scenef in scenefs:
        scenef = np.reshape(scenef, 2048)
        print(scenef.shape)
        # (2048,) --> (1, 2048) --> (16, 128)
    #    scenef = scenef.reshape(-1, 128)
    #    print(scenef.shape)
        new_scenefs.append(scenef)
    for objf in objfs:
	objf = np.reshape(objf,200704)
	new_objfs.append(objf)
   # print(new_scenefs[0].shape)
   # print(scenefs[0].shape)

    #load histograms
#    with open("discogs_hot_histos.pickle", 'rb') as f:
#        pkl = pickle.load(f)
#        img_names = pkl[0]
#        histos = pkl[1]
#    print(len(histos))
#    new_histos = []
#    for histo in histos:
#        histo = np.reshape(histo,46080)
#	print(histo.shape)
#	new_histos.append(histo)

    
    descriptors=list()
    idImage =list()
    for imagePath in glob.glob(path+"/*.jpg"):
        print(imagePath)
	img = cv2.imread(imagePath)

#        scenef = new_scenefs[img_names.index(imagePath.split("/")[-1])]
	objf = new_objfs[img_names.index(imagePath.split("/")[-1])]
#	print(scenef.shape)

#	histo = new_histos[img_names.index(imagePath.split("/")[-1])]
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(img,None)
	if np.any(des)!=None:
            v=VLAD(des,visualDictionary)
	    #print(v.shape)
#	    mergedf =  np.concatenate([v, scenef]).reshape(-1,256)
	    mergedf = np.concatenate([v, objf])
#	    mergedf = np.concatenate([v, scenef])
#	    mergedf = np.concatenate([v,histo])
	    print(mergedf.shape)

#            pca = PCA(n_components=256)
#            pcaf = pca.fit_transform(mergedf)
#	    pcaf = pcaf.flatten()
#            print(pcaf.shape)

#            descriptors.append(pcaf)
            descriptors.append(mergedf)
            idImage.append(imagePath)
                    
    descriptors = np.asarray(descriptors)
    print(descriptors.shape)
    return descriptors, idImage


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





ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required = True,
	help = "Path to where VLAD descriptors will be stored")
args = vars(ap.parse_args())

path = "/home/comp/e4252392/discogs_hot"
pathVD = "/home/comp/e4252392/VLAD-master/discogs_hot_vd.pickle"
output=args["output"]

print("estimating VLAD descriptors using SIFT  for dataset: /" + path + " and visual dictionary: /" + pathVD)
with open(pathVD, 'rb') as f:
    visualDictionary = pickle.load(f) 

V, idImages = getVLADDescriptors(path,visualDictionary)
print(V.shape)

file = output + ".pickle"
print("Dumping...")
with open(file, 'wb') as f:
	pickle.dump([idImages,V,path], f)
print("Thank God writen to file \m/")
print("The HISTO descriptors are saved in "+file)

