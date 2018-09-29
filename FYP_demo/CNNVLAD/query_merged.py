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
import itertools
from sklearn.decomposition import PCA
import keras_frcnn.resnet as nn
from sklearn.cross_decomposition import CCA
from keras.layers import Input, Add, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, TimeDistributed

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

    # pca = PCA(n_components=32)
    # pcav = pca.fit_transform(V.reshape(256,256)).flatten()
    # V = pcav
    # print("pcav.shape = ")
    # print(pcav.shape)
    V = V.flatten()
    V = np.sign(V) * np.sqrt(np.abs(V))
    V = V / np.sqrt(np.dot(V, V))
    # V = V.reshape(512, -1) # change 128 to change remained CCA components
    return V


parser = OptionParser()
parser.add_option("-r", "--retrieve", dest="num2retrieve", default='10')
parser.add_option("--vd", dest="pathVD", default="/users/sunjingxuan/pycharmprojects/CNNVLAD/vd_logos_cpu3.pickle")
parser.add_option("--vdcnn", dest="pathColf", default="/users/sunjingxuan/pycharmprojects/CNNVLAD/vdcnn_logos_preres_600.pickle")
parser.add_option("-i", "--index", dest="treeIndex", default="/users/sunjingxuan/pycharmprojects/CNNVLAD/tree_vladcnn_logos_preres_600.pickle")
parser.add_option("--config", dest="config", default="/users/sunjingxuan/desktop/frcnn-original-weights/config.pickle")
parser.add_option("--model_path", dest="model_path", default='placeholder')
parser.add_option("-p", "--image_path", dest="image_path", default="/users/sunjingxuan/desktop/matchlogo")
(options, args) = parser.parse_args()

# for VLAD
k = int(options.num2retrieve)
pathVD = options.pathVD
treeIndex = options.treeIndex
pathColf = options.pathColf

print('Load VLAD docs: ' + pathVD + ', ' + treeIndex)
with open(treeIndex, 'rb') as f:
    indexStructure = pickle.load(f)
with open(pathVD, 'rb') as f:
    visualDictionary = pickle.load(f)
with open(pathColf, 'rb') as f:
    vd_colf = pickle.load(f)

imageID = indexStructure[0]
tree = indexStructure[1]
# pathImageData = indexStructure[2]
# print(pathImageData) #/home/comp/e4252392/discogs_hot
print('VLAD docs loaded.')


#for CNN
print('Preparing CNN config...')
config_output_filename = options.config
with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

input_shape_img = (None, None, 3)
# input_shape_img = (224, 224, 3)
img_input = Input(shape=input_shape_img)
shared_layers = nn.nn_base(img_input, trainable=False)
x = conv_block(shared_layers, 3, [512, 512, 2048], stage=5, block='a')
x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
x = AveragePooling2D((7, 7), name='avg_pool')(x)
scene_extractor = Model(img_input, x)
# x = Flatten()(x)
# x = Dense(256, activation='relu')(x)
# x = Dense(1000, activation='softmax', name='fc1000')(x)
# global_extractor = Model(img_input, x)

model_path = options.model_path
# pre_weight_path = "/users/sunjingxuan/desktop/frcnn-original-weights/model_frcnn.hdf5"
# pre_weight_path = '/users/sunjingxuan/desktop/second_res_more_epoch.h5'
pre_weight_path = '/users/sunjingxuan/desktop/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
if model_path is not 'placeholder':
    final_weights = model_path
else:
    final_weights = pre_weight_path
print('Loading CNN weights from {}'.format(final_weights))
scene_extractor.load_weights(final_weights, by_name=True)
# global_extractor.load_weights(final_weights, by_name=True)
print('CNN weights loaded.')

img_names = []
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
        # img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, descriptor = sift.detectAndCompute(img, None)

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
        scenef = scene_extractor.predict(X)
        print("scenef.shape = ")
        print(scenef.shape)
        # fc = global_extractor.predict(X)[0]
        # print("fc.shape = ")
        # print(fc.shape)
        # if '127696' in img_name:
            # print(fc)

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
        # print(scenef.shape) 


        colf = []
        scenef = scenef[0]
        rows = scenef.shape[0]
        columns = scenef.shape[1]
        for i in range(rows):
            for j in range(columns):
                colf.append(scenef[i,j])
        colf = np.asarray(colf)
        print(colf.shape)

        # calc VLAD
        print('Calc VLAD...')
        v = VLAD(descriptor, visualDictionary)
        vlad_colf = VLAD(colf, vd_colf)

        # print("==========Performing CCA==========")
        # cca = CCA(n_components=1)
        # v_c, vlad_colf_c = cca.fit_transform(v, vlad_colf)
        # mergedf = np.concatenate([v_c, vlad_colf_c])
        # mergedf = mergedf.reshape(1, -1)[0]
        # print("mergedf.shape = ")
        # print(mergedf.shape)
        # print("==================================")
        
        # mergedf = np.concatenate([v, scenef.flatten()])
        mergedf = np.concatenate([v, vlad_colf])

    except cv2.error:
        print("OpeCV Error: Bad argument (image is empty or has incorrect depth! but pass.")


    print('Searching tree...')
    dist = []
    ind = []
    # if np.any(np.isnan(v)):
    # dist, ind = tree.query(v.reshape(1, -1), k)
    dist, ind = tree.query(mergedf.reshape(1, -1), k)
    # dist, ind = tree.query(scenef.flatten().reshape(1, -1), k)
    # dist, ind = tree.query(fc.reshape(1, -1), k)
    print(dist)
    print(ind)
    if len(dist) != 0 and len(ind) != 0:
        ind = list(itertools.chain.from_iterable(ind))

        print(filepath)
        for i in ind:
            print(imageID[i])


