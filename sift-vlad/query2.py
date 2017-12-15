#merged query procedure
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

    V = V.flatten()
    V = np.sign(V) * np.sqrt(np.abs(V))
    V = V / np.sqrt(np.dot(V, V))
    return V


parser = OptionParser()
parser.add_option("--config_filename", dest="config_filename", default="config5.pickle")
parser.add_option("--network", dest="network", default='resnet50')

parser.add_option("-r", "--retrieve", dest="num2retrieve", default='5')
parser.add_option("-d", "--visualDictionary", dest="pathVD", default="discogs_hot_vd.pickle")
parser.add_option("-i", "--index", dest="treeIndex", default="discogs_hot_tree_merged.pickle")
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
config_output_filename = options.config_filename
with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

if C.network == 'resnet50':
    import keras_frcnn.resnet as nn
elif C.network == 'vgg':
    import keras_frcnn.vgg as nn

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

img_path = "/home/comp/e4252392/VLAD-master/matchtest1111"
if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
else:
    input_shape_img = (224, 224, 3)
img_input = Input(shape=input_shape_img)

shared_layers = nn.nn_base(img_input, trainable=False)
obj_extractor = Model(img_input, shared_layers)

x = conv_block(shared_layers, 3, [512, 512, 2048], stage=5, block='a')
x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
x = AveragePooling2D((7, 7), name='avg_pool')(x)
scene_extractor = Model(img_input, x)

#print('Loading CNN weights from {}'.format(C.model_path))
#obj_extractor.load_weights(C.model_path, by_name=True)
#scene_extractor.load_weights(C.model_path, by_name=True)

pre_weight_path = "/home/comp/e4252392/cnntrees/resnet50_weights_tf_dim_ordering_tf_kernels.h5"
print('Loading CNN weights from {}'.format(pre_weight_path))
obj_extractor.load_weights(pre_weight_path, by_name=True)
scene_extractor.load_weights(pre_weight_path, by_name=True)
print('CNN weights loaded.')

# objfs = []
# scenefs = []
img_names = []
visualise = True

for idx, img_name in enumerate(sorted(os.listdir(img_path))):
    if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
        continue
    print(img_name)
    st = time.time()
    filepath = os.path.join(img_path, img_name)
    img = cv2.imread(filepath)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

    #calc VLAD
    print('Calc VLAD...')
    sift = cv2.xfeatures2d.SIFT_create()
    kp, descriptor = sift.detectAndCompute(img, None)
    v = VLAD(descriptor, visualDictionary)

    #calc histo
#    print('Calc Histo...')
#    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
#    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
#    hist = np.reshape(hist,46080)


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
    objf = obj_extractor.predict(X)
    # objfs.append(objf)
    scenef = scene_extractor.predict(X)
    # scenefs.append(scenef)

# objfs = np.asanyarray(objfs, dtype=object)


    print('Merge VLAD and CNN features...')
#    scenef = np.reshape(scenef, 2048) #(2048,)
    objf = np.reshape(objf,200704)
#    mergedf = np.concatenate([objf,scenef])
    mergedf = np.concatenate([v, objf])
#    mergedf = np.concatenate([v,hist]) 
#    print(mergedf.shape)

#    mergedf = mergedf.reshape(-1, 256)
#    pca = PCA(n_components=256)
#    pcaf = pca.fit_transform(mergedf).flatten()
#    print(pcaf.shape)


    print('Searching tree...')
#    dist, ind = tree.query(scenef.reshape(1, -1), k)
#    dist, ind = tree.query(pcaf.reshape(1, -1), k)
    dist, ind = tree.query(mergedf.reshape(1, -1), k)
#    dist, ind = tree.query(objf.reshape(1, -1), k)
#    dist, ind = tree.query(hist.reshape(1, -1), k)
    print(dist)
    print(ind)
    ind = list(itertools.chain.from_iterable(ind))

    print(filepath)
    for i in ind:
        print(imageID[i])


