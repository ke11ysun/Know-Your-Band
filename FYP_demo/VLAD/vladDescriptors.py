import argparse
import glob
import cv2
import numpy as np
import pickle
from sklearn.decomposition import PCA

def getVLADDescriptors(path, pathVD, pathCNNGT):
    with open(pathVD, 'rb') as f:
        visualDictionary = pickle.load(f)
    # load cnn features
    with open(pathCNNGT, 'rb') as f:
        pkl = pickle.load(f)
        # objfs = pkl[1]
        # scenefs = pkl[2]
        scenefs = pkl[1]
        img_names = pkl[0]

    # load histograms
    # with open("discogs_hot_histos.pickle", 'rb') as f:
    # pkl = pickle.load(f)
    # img_names = pkl[0]
    # histos = pkl[1]
    # print(len(histos))
    # new_histos = []
    # for histo in histos:
    #   histo = np.reshape(histo,46080)
    #	print(histo.shape)
    #	new_histos.append(histo)

    descriptors = list()
    idImage = list()
    for imagePath in glob.glob(path + "/*.jpg"):
        print(imagePath)
        img = cv2.imread(imagePath)
        print(img_names.index(imagePath.split("/")[-1]))
        scenef = scenefs[img_names.index(imagePath.split("/")[-1])]
        print(scenef.shape)
        if scenef.shape[1]>2:
            scenef = scenef[:,:2,:,:]
        if scenef.shape[1]<2:
            npad = ((0, 0), (0, 2-scenef.shape[1]), (0, 0), (0, 0))
            scenef = np.pad(scenef, pad_width=npad, mode='constant', constant_values=0)
        if scenef.shape[2]>7:
            scenef = scenef[:,:,:7,:]
        if scenef.shape[2]<7:
            npad = ((0, 0), (0, 0), (0, 7-scenef.shape[2]), (0, 0))
            scenef = np.pad(scenef, pad_width=npad, mode='constant', constant_values=0)
        # objf = objfs[img_names.index(imagePath.split("/")[-1])]
        # print(objf.shape)
        # if objf.shape[1]>38:
        #     objf = objf[:,:38,:,:]
        # if objf.shape[1]<38:
        #     npad = ((0, 0), (0, 38-objf.shape[1]), (0, 0), (0, 0))
        #     objf = np.pad(objf, pad_width=npad, mode='constant', constant_values=0)
        # if objf.shape[2]>103:
        #     objf = objf[:,:,:103,:]
        # if objf.shape[2]<103:
        #     npad = ((0, 0), (0, 0), (0, 103-objf.shape[2]), (0, 0))
        #     objf = np.pad(objf, pad_width=npad, mode='constant', constant_values=0)
	# print(scenef.shape)

        # histo = new_histos[img_names.index(imagePath.split("/")[-1])]
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
        if np.any(des) != None:
            v = VLAD(des, visualDictionary)
            # print(v.shape)
            mergedf = scenef.flatten()
            # print(mergedf.shape)
            # mergedf = np.concatenate([scenef, objf])
            # mergedf = np.concatenate([v, objf])
            # mergedf = np.concatenate([v, scenef.flatten()])
            # mergedf = np.concatenate([v,histo])
            # print(mergedf.shape)

            # descriptors.append(v)
            descriptors.append(mergedf)
            idImage.append(imagePath)

    descriptors = np.asarray(descriptors)
    print(descriptors.shape)
    return descriptors, idImage


def VLAD(X, visualDictionary):
    predictedLabels = visualDictionary.predict(X)
    centers = visualDictionary.cluster_centers_
    labels = visualDictionary.labels_
    k = visualDictionary.n_clusters

    m, d = X.shape
    V = np.zeros([k, d])
    for i in range(k):
        if np.sum(predictedLabels == i) > 0:
            V[i] = np.sum(X[predictedLabels == i, :] - centers[i], axis=0)

    # pca = PCA(n_components=25)
    # pcav = pca.fit_transform(V.reshape(256,256)).flatten()
    # print("pcav.shape = ")
    # print(pcav.shape)
    V = V.flatten()
    # V = pcav
    V = np.sign(V) * np.sqrt(np.abs(V))
    V = V / np.sqrt(np.dot(V, V))
    return V


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="Path to where VLAD descriptors will be stored")
ap.add_argument("-p", "--image_path", required=True)
ap.add_argument("--vd",required=True)
ap.add_argument("--cnngt", default='placeholder')
args = vars(ap.parse_args())

output = args["output"]
path = args["image_path"]
pathVD = args["vd"]
pathCNNGT = args["cnngt"]

print("estimating VLAD descriptors using SIFT  for dataset: /" + path + " and visual dictionary: /" + pathVD)
V, idImages = getVLADDescriptors(path, pathVD, pathCNNGT)
print(V.shape)

file = output + ".pickle"
print("Dumping...")
with open(file, 'wb') as f:
    pickle.dump([idImages, V, path], f)
print("Thank God written to file \m/")
print("The VLAD descriptors are saved in " + file)
