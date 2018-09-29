from __future__ import division
import os
import numpy as np
import pickle
from optparse import OptionParser
import time
from keras import backend as K
from keras.models import Model
import cv2
from keras_frcnn.resnet import identity_block, conv_block
from keras.layers import Input, AveragePooling2D
import itertools
from sklearn.decomposition import PCA
import keras_frcnn.resnet as nn

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


parser = OptionParser()
parser.add_option("-r", "--retrieve", dest="num2retrieve", default='5')
parser.add_option("--vd", dest="pathVD", default="discogs_hot_vd.pickle")
parser.add_option("-i", "--index", dest="treeIndex", default="discogs_hot_tree_merged.pickle")
parser.add_option("--config", dest="config", default="/users/sunjingxuan/desktop/frcnn-original-weights/config.pickle")
parser.add_option("--model_path", dest="model_path", default='placeholder')
parser.add_option("-p", "--image_path", dest="image_path")
(options, args) = parser.parse_args()

#for VLAD
k = int(options.num2retrieve)
pathVD = options.pathVD
treeIndex = options.treeIndex

print('Load VLAD docs: ' + pathVD + ', ' + treeIndex)
with open(treeIndex, 'rb') as f:
    indexStructure = pickle.load(f)
with open(pathVD, 'rb') as f:
    visualDictionary = pickle.load(f)

imageID = indexStructure[0]
tree = indexStructure[1]
pathImageData = indexStructure[2]
print(pathImageData) #/home/comp/e4252392/discogs_hot
print('VLAD docs loaded.')


#for CNN
print('Preparing CNN config...')
config_output_filename = options.config
with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
else:
    input_shape_img = (None, None, 3)
img_input = Input(shape=input_shape_img)

shared_layers = nn.nn_base(img_input, trainable=False)
# obj_extractor = Model(img_input, shared_layers)

x = conv_block(shared_layers, 3, [512, 512, 2048], stage=5, block='a')
x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
x = AveragePooling2D((7, 7), name='avg_pool')(x)
scene_extractor = Model(img_input, x)

model_path = options.model_path
pre_weight_path = "/users/sunjingxuan/desktop/frcnn-original-weights/model_frcnn.hdf5"
# pre_weight_path = "/users/sunjingxuan/pycharmprojects/cnntest/resnet50_weights_tf_dim_ordering_tf_kernels.h5"
if model_path is not 'placeholder':
    final_weights = model_path
else:
    final_weights = pre_weight_path
print('Loading CNN weights from {}'.format(final_weights))
# obj_extractor.load_weights(final_weights, by_name=True)
scene_extractor.load_weights(final_weights, by_name=True)
print('CNN weights loaded.')

img_names = []
visualise = True
img_path = options.image_path
for idx, img_name in enumerate(sorted(os.listdir(img_path))):
    if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
        continue
    print(img_name)
    st = time.time()
    filepath = os.path.join(img_path, img_name)
    img = cv2.imread(filepath)
    v = np.nan

    try:
        img_min_side = float(C.im_size)
        (height, width, _) = img.shape
        if width <= height:
            ratio = img_min_side / width
            new_height = int(ratio * height)
            new_width = int(img_min_side)
        else:
            ratio = img_min_side / height
            new_width = int(ratio * width)
            new_height = int(img_min_side)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        #img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        #calc VLAD
        print('Calc VLAD...')
        sift = cv2.xfeatures2d.SIFT_create()
        kp, descriptor = sift.detectAndCompute(img, None)
        #print(type(descriptor))
        #print(np.any(np.isnan(descriptor)))
        v = VLAD(descriptor, visualDictionary)
        # calc histo
        # print('Calc Histo...')
        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        # hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        # hist = np.reshape(hist,46080)

        #calc CNN
        print('Calc CNN...')
        img = img[:, :, (2, 1, 0)]
        img = img.astype(np.float32)
        img[:, :, 0] -= C.img_channel_mean[0]
        img[:, :, 1] -= C.img_channel_mean[1]
        img[:, :, 2] -= C.img_channel_mean[2]
        img /= C.img_scaling_factor
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        X = img
        if K.image_dim_ordering() == 'tf':
            X = np.transpose(X, (0, 2, 3, 1))

        img_names.append(img_name)
        # objf = obj_extractor.predict(X)
        scenef = scene_extractor.predict(X)
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
        print(scenef.shape) 
        # if objf.shape[1]>38:
        #     objf = objf[:,:38,:,:]
        # if objf.shape[1]<38:
        #     npad = ((0, 0), (0, 38-objf.shape[1]), (0, 0), (0, 0))
        #     objf = np.pad(objf, pad_width=npad, mode='constant', constant_values=0)
        # if objf.shape[2]>103:
        #     objf = objf[:,:,:103,:]
        # if scenef.shape[2]<103:
        #     npad = ((0, 0), (0, 0), (0, 103-objf.shape[2]), (0, 0))
        #     objf = np.pad(objf, pad_width=npad, mode='constant', constant_values=0)
        # print(objf.shape)
        # print('Merge VLAD and CNN features...')
        # scenef = np.reshape(scenef, 2048)
        # objf = np.reshape(objf,200704)
        # mergedf = np.concatenate([objf,scenef])
        # mergedf = np.concatenate([v, scenef.flatten()])
        #print(mergedf.shape)

    except cv2.error:
        print("OpeCV Error: Bad argument (image is empty or has incorrect depth! but pass.")
    # except ValueError:
    #     print("ValueError: nan! but pass.")

    print('Searching tree...')
    dist = []
    ind = []
    #if np.any(np.isnan(v)):
    # dist, ind = tree.query(v.reshape(1, -1), k)
    # dist, ind = tree.query(mergedf.reshape(1, -1), k)
    dist, ind = tree.query(scenef.flatten().reshape(1, -1), k)
    # dist, ind = tree.query(objf.reshape(1, -1), k)
    # dist, ind = tree.query(hist.reshape(1, -1), k)
    print(dist)
    print(ind)
    if len(dist) != 0 and len(ind) != 0:
        ind = list(itertools.chain.from_iterable(ind))

        print(filepath)
        for i in ind:
            print(imageID[i])


