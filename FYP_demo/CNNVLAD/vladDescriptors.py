import argparse
import glob
import cv2
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

def getVLADDescriptors(path, pathVD, pathCNNGT, pathColf):
    with open(pathVD, 'rb') as f:
        visualDictionary = pickle.load(f)
    # load cnn features
    with open(pathCNNGT, 'rb') as f:
        pkl = pickle.load(f)
        # fcs = pkl[2]
        scenefs = pkl[1]
        img_names = pkl[0]
    

    # update 3/22 VLAD on column feature 
    with open(pathColf, 'rb') as f:
        vd_colf = pickle.load(f)


    descriptors = list()
    idImage = list()
    for imagePath in glob.glob(path + "/*.jpg"):
        print(imagePath)
        img = cv2.imread(imagePath)
        print(img_names.index(imagePath.split("/")[-1]))
        scenef = scenefs[img_names.index(imagePath.split("/")[-1])]
        print("scenef.shape = ")
        print(scenef.shape)
        # fc = fcs[img_names.index(imagePath.split("/")[-1])][0]
        # print("fc.shape = ")
        # print(fc.shape)

        # if scenef.shape[1]>2:
        #     scenef = scenef[:,:2,:,:]
        # if scenef.shape[1]<2:
        #     npad = ((0, 0), (0, 2-scenef.shape[1]), (0, 0), (0, 0))
        #     scenef = np.pad(scenef, pad_width=npad, mode='constant', constant_values=0)
        # if scenef.shape[2]>7:
        #     scenef = scenef[:,:,:7,:]
        # if scenef.shape[2]<7:
        #     npad = ((0, 0), (0, 0), (0, 7-scenef.shape[2]), (0, 0))
        #     scenef = np.pad(scenef, pad_width=npad, mode='constant', constant_values=0)


        colf = []
        scenef = scenef[0]
        rows = scenef.shape[0]
        columns = scenef.shape[1]
        for i in range(rows):
            for j in range(columns):
                colf.append(scenef[i,j])
        colf = np.asarray(colf)
        print(colf.shape)

            

        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
        if np.any(des) != None:  # and np.any(colf) != None
            v = VLAD(des, visualDictionary)
            vlad_colf = VLAD(colf, vd_colf)

            # mergedf = scenef.flatten()
            # mergedf = np.concatenate([v, scenef.flatten()])
            # mergedf = np.concatenate([v, vlad_colf])
            # mergedf = np.concatenate([v, fc])
            # print("mergedf.shape = ")
            # print(mergedf.shape)

            print("==========Performing CCA==========")
            cca = CCA(n_components=1)
            v_c, vlad_colf_c = cca.fit_transform(v, vlad_colf)
            # print(v_c)
            print(v_c.shape)
            # print(vlad_colf_c)
            print(vlad_colf_c.shape)
            mergedf = np.concatenate([v_c, vlad_colf_c])
            mergedf = mergedf.reshape(1, -1)[0]
            print("mergedf.shape = ")
            print(mergedf.shape)
            print("==================================")


            # descriptors.append(fc)
            # if '127696' in imagePath:
                # print(fc)
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

    pca = PCA(n_components=32)
    pcav = pca.fit_transform(V.reshape(256,256)).flatten()
    print("pcav.shape = ")
    print(pcav.shape)
    
    V = pcav
    # V = V.flatten()
    V = np.sign(V) * np.sqrt(np.abs(V)) # why is this even used for l2 norm???
    V = V / np.sqrt(np.dot(V, V))
    # print(np.isnan(np.min(V)))
    # print(np.isinf(np.max(V)))

    V = V.reshape(512, -1) # change 128 to change remained CCA components
    print("V.shape = ")
    print(V.shape)
    return V


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True)
ap.add_argument("-p", "--image_path", required=True)
ap.add_argument("--vd",required=True)
ap.add_argument("--cnngt", default='placeholder')
ap.add_argument("--vdcnn", required=True)
args = vars(ap.parse_args())

output = args["output"]
path = args["image_path"]
pathVD = args["vd"]
pathCNNGT = args["cnngt"]
pathColf = args["vdcnn"]
# pathColf = 'ajajajaj'

print("estimating VLAD descriptors using SIFT/Column feature for dataset: /" + path + " and visual dictionary: /" + pathVD)
V, idImages = getVLADDescriptors(path, pathVD, pathCNNGT, pathColf)
print(V.shape)

file = output + ".pickle"
print("Dumping...")
with open(file, 'wb') as f:
    pickle.dump([idImages, V, path], f)
print("Thank God written to file \m/")
print("The VLAD descriptors are saved in " + file)
