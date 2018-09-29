import argparse
import glob
import numpy as np
import pickle

def getVLADDescriptors(path, pathVD, pathCNNGT):
    with open(pathVD, 'rb') as f:
        visualDictionary = pickle.load(f)
    # load cnn features
    with open(pathCNNGT, 'rb') as f:
        pkl = pickle.load(f)
        objfs = pkl[1]
        scenefs = pkl[2]
        img_names = pkl[0]

    # row_objfs = []
    # col_objfs = []
    new_scenefs = []
    # row_scenefs = []
    # col_scenefs = []
    new_objfs = []
    for scenef in scenefs:
        # print(scenef.shape)
        #scenef = np.reshape(scenef, 2048)
        # row_scenefs.append(scenef.shape[1])
        # col_scenefs.append(scenef.shape[2])
        if scenef.shape[1]>2:
            scenef = scenef[:,:2,:,:]
        if scenef.shape[1]<2:
            npad = ((0, 0), (0, 2-scenef.shape[1]), (0, 0), (0, 0))
            scenef = np.pad(scenef, pad_width=npad, mode='constant', constant_values=0)
        if scenef.shape[2]>2:
            scenef = scenef[:,:,:2,:]
        if scenef.shape[2]<2:
            npad = ((0, 0), (0, 0), (0, 2-scenef.shape[2]), (0, 0))
            scenef = np.pad(scenef, pad_width=npad, mode='constant', constant_values=0)
        print(scenef.shape)  

    for objf in objfs:
        # print(objf.shape)
        #objf = np.reshape(objf, 200704)
        # row_objfs.append(objf.shape[1])        
        # col_objfs.append(objf.shape[2])
        if objf.shape[1]>38:
            objf = objf[:,:38,:,:]
        if objf.shape[1]<38:
            npad = ((0, 0), (0, 38-objf.shape[1]), (0, 0), (0, 0))
            objf = np.pad(objf, pad_width=npad, mode='constant', constant_values=0)
        if objf.shape[2]>103:
            objf = objf[:,:,:103,:]
        if objf.shape[2]<103:
            npad = ((0, 0), (0, 0), (0, 103-objf.shape[2]), (0, 0))
            objf = np.pad(objf, pad_width=npad, mode='constant', constant_values=0)
        print(objf.shape)
    # print('scenef shape 1: ' + str(np.mean(row_scenefs)))
    # print('scenef shape 2: ' + str(np.mean(col_scenefs)))
    # print('objf shape 1: ' + str(np.mean(row_objfs)))
    # print('objf shape 2: ' + str(np.mean(col_objfs)))


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
getVLADDescriptors(path, pathVD, pathCNNGT)

