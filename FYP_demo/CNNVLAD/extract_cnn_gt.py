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
import keras_frcnn.resnet as nn

sys.setrecursionlimit(40000)
parser = OptionParser()
parser.add_option("-p", "--path", dest="gt_path")
parser.add_option("--config", dest="config_filename", default='/users/sunjingxuan/desktop/frcnn-original-weights/config.pickle')
parser.add_option("--model_path", dest="model_path", default='/users/sunjingxuan/desktop/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
parser.add_option("--output", dest="output")
(options, args) = parser.parse_args()

if not options.gt_path:
    parser.error('Error: path to test data')
if not options.model_path:
    parser.error('Error: path to model weights')

config_output_filename = options.config_filename
with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

input_shape_img = (None, None, 3)
# input_shape_img = (224, 224, 3)
img_input = Input(shape=input_shape_img)

shared_layers = nn.nn_base(img_input, trainable=False)

# classifier_layers
x = conv_block(shared_layers, 3, [512, 512, 2048], stage=5, block='a')
x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
x = AveragePooling2D((7, 7), name='avg_pool')(x)
scene_extractor = Model(img_input, x)

'''USE FC FEATURE'''
# x = Flatten()(x)
# # x = Dense(256, activation='relu')(x)
# x = Dense(1000, activation='softmax', name='fc1000')(x)
# global_extractor = Model(img_input, x)

model_path = options.model_path
print('Loading weights from {}'.format(model_path))
scene_extractor.load_weights(model_path, by_name=True)
# global_extractor.load_weights(model_path, by_name=True)
# global_extractor.summary()

img_path = options.gt_path
scenefs = []
fcs = []
img_names = []

for idx, img_name in enumerate(sorted(os.listdir(img_path))):
    if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
        continue
    print(img_name)
    st = time.time()
    filepath = os.path.join(img_path,img_name)
    img = cv2.imread(filepath)

    # X, ratio = test_frcnn.format_img(img, C)
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
    print(scenef.shape)     # (1, 3, 2, 2048)
    scenefs.append(scenef)
    # fc = global_extractor.predict(X)
    # print(fc.shape)
    # fcs.append(fc)

# output
outfile = options.output
with open(outfile, 'wb') as f:
    pickle.dump([img_names, scenefs], f, protocol=2)
print("The ground truth features are saved in " + outfile)


