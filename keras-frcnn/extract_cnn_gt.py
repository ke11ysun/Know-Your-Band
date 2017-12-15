from __future__ import division
import os
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
import cv2
from keras_frcnn.resnet import identity_block, conv_block, classifier_layers
from keras.layers import Input, Add, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, TimeDistributed


sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="gt_path", help="Path to ground truth images.")
parser.add_option("--config_filename", dest="config_filename", help=
                "Location to read the metadata related to the training (generated when training).",
                default="config5.pickle")
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')

(options, args) = parser.parse_args()

if not options.gt_path:   # if filename is not given
    parser.error('Error: path to test data must be specified. Pass --path to command line')

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

img_path = options.gt_path

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
else:
    input_shape_img = (224, 224, 3)

img_input = Input(shape=input_shape_img)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=False)
obj_extractor = Model(img_input, shared_layers)

# x = classifier_layers(shared_layers, input_shape=(14,14,1024), trainable=False)
x = conv_block(shared_layers, 3, [512, 512, 2048], stage=5, block='a')
x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
x = AveragePooling2D((7, 7), name='avg_pool')(x)
# print(x.shape)  (?, ?, ?, 2048)
# x = Flatten()(x)
scene_extractor = Model(img_input, x)

print('Loading weights from {}'.format(C.model_path))
obj_extractor.load_weights(C.model_path, by_name=True)
scene_extractor.load_weights(C.model_path, by_name=True)

objfs = []
scenefs = []
# gts = []
# [img_name, objf, scenef]
img_names = []
visualise = True

for idx, img_name in enumerate(sorted(os.listdir(img_path))):
    if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
        continue
    print(img_name)
    st = time.time()
    filepath = os.path.join(img_path,img_name)

    img = cv2.imread(filepath)

    # # X, ratio = test_frcnn.format_img(img, C)
    # img_min_side = float(C.im_size)
    # (height, width, _) = img.shape
    #
    # if width <= height:
    #     ratio = img_min_side / width
    #     new_height = int(ratio * height)
    #     new_width = int(img_min_side)
    # else:
    #     ratio = img_min_side / height
    #     new_width = int(ratio * width)
    #     new_height = int(img_min_side)
    # img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
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
    # print(objf.shape)
    # (1, 14, 14, 1024)
    objfs.append(objf)
    # scenef = Flatten()(K.variable(scene_extractor.predict(X)))
    scenef = scene_extractor.predict(X)
    # print(scenef.shape)
    # (1, 3, 2, 2048)
    scenefs.append(scenef)
    # gts.append([img_name, objf, scenef])


# objfs = np.asanyarray(objfs, dtype=object)

# output
file = "discogs_hot_cnngt.pickle"
with open(file, 'wb') as f:
    pickle.dump([img_names, objfs, scenefs], f, protocol=2)
print("The ground truth features are saved in " + file)



# with open("discogs_hot_cnngt.pickle", 'rb') as f:
#     pkl = pickle.load(f)
#     scenefs = pkl[2]
#     img_names = pkl[0]
# new_scenefs = []
# for scenef in scenefs:
#     scenef = np.reshape(scenef, 2048)
#     print(scenef.shape)
#     # (2048,) --> (1, 2048) --> (16, 128)
#     scenef = scenef.reshape(-1,128)
#     print(scenef.shape)
#     new_scenefs.append(scenef)
#
# print(new_scenefs[0].shape)
# print(type(new_scenefs[img_names.index("R-498424-1196732600.jpeg.jpg")]))

    # scenef = np.reshape(scenefs[0], 2048)
    # print(scenef.shape)
    # # (2048,) --> (1, 2048) --> (16, 128)
    # scenef = scenef.reshape(-1,128)
    # print(scenef.shape)

