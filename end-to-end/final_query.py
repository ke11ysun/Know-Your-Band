from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers



def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height,width,_) = img.shape

    if width <= height:
        ratio = img_min_side/width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side/height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio

def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2 ,real_y2)


def test_frcnn(path):
    from keras.models import Model
    from keras_frcnn.RoiPoolingConv import RoiPoolingConv

    sys.setrecursionlimit(40000)
    parser = OptionParser()
    parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
    parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
                    help="Number of ROIs per iteration. Higher means more memory use.", default=32)
    parser.add_option("--config_filename", dest="config_filename", help=
                    "Location to read the metadata related to the training (generated when training).",
                    default="config5.pickle")
    parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')
    (options, args) = parser.parse_args()

    # if not options.test_path:   # if filename is not given
    #     parser.error('Error: path to test data must be specified. Pass --path to command line')

    config_output_filename = options.config_filename

    with open(config_output_filename, 'rb') as f_in:
        C = pickle.load(f_in)

    if C.network == 'resnet50':
        import keras_frcnn.resnet as nn
    elif C.network == 'vgg':
        import keras_frcnn.vgg as nn

    C.use_horizontal_flips = False
    C.use_vertical_flips = False
    C.rot_90 = False

    # img_path = options.test_path
    img_path = path
    class_mapping = C.class_mapping

    if 'bg' not in class_mapping:
        class_mapping['bg'] = len(class_mapping)

    class_mapping = {v: k for k, v in class_mapping.items()}
    print(class_mapping)
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
    C.num_rois = int(options.num_rois)

    if C.network == 'resnet50':
        num_features = 1024
    elif C.network == 'vgg':
        num_features = 512

    if K.image_dim_ordering() == 'th':
        input_shape_img = (3, None, None)
        input_shape_features = (num_features, None, None)
    else:
        input_shape_img = (None, None, 3)
        input_shape_features = (None, None, num_features)


    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(C.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    shared_layers = nn.nn_base(img_input, trainable=True)

    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)

    classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    model_classifier = Model([feature_map_input, roi_input], classifier)

    print('Loading weights from {}'.format(C.model_path))
    model_rpn.load_weights(C.model_path, by_name=True)
    model_classifier.load_weights(C.model_path, by_name=True)

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    all_imgs = []
    classes = {}
    bbox_threshold = 0.8
    visualise = True
    crops = []

    for idx, img_name in enumerate(sorted(os.listdir(img_path))):
        if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            continue
        print(img_name)
        st = time.time()
        filepath = os.path.join(img_path,img_name)

        img = cv2.imread(filepath)

        X, ratio = format_img(img, C)

        if K.image_dim_ordering() == 'tf':
            X = np.transpose(X, (0, 2, 3, 1))

        [Y1, Y2, F] = model_rpn.predict(X)
        R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        bboxes = {}
        probs = {}
        roifs = {}
        scenefs = {}

        for jk in range(R.shape[0]//C.num_rois + 1):
            ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0]//C.num_rois:                #pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            out_roi_pool = RoiPoolingConv(14, C.num_rois)([feature_map_input, roi_input])
            haha = Model(inputs=[feature_map_input, roi_input], outputs=out_roi_pool)

            all_roifs = haha.predict([F, ROIs])
            [all_scenefs, P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

            for ii in range(P_cls.shape[1]):
                if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []
                    roifs[cls_name] = []
                    scenefs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))
                roifs[cls_name].append(all_roifs[0, ii, :])
                scenefs[cls_name].append(all_scenefs[0, ii, :])

        all_dets = []

        for key in bboxes:
            bbox = np.array(bboxes[key])
            prob = np.array(probs[key])
            roif = np.array(roifs[key])
            scenef = np.array(scenefs[key])

            new_boxes, new_probs, new_roifs, new_scenefs = roi_helpers.nmsf(bbox, prob, roif, scenef, overlap_thresh=0.5)
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk,:]

                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

                cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

                textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
                # all_dets.append((key,100*new_probs[jk], new_roifs[jk].shape, new_scenefs[jk].shape))
                all_dets.append((key,100*new_probs[jk],real_x1,real_x2,real_y1,real_y2))

                (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
                textOrg = (real_x1, real_y1-0)

                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
                cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
                # np.save('new_roifs.npy', new_roifs)
                # np.save('new_scenefs.npy', new_scenefs)

        for i in range(len(all_dets)):
            cropped = img[all_dets[i][4]:all_dets[i][5], max(0, all_dets[i][2]):all_dets[i][3]]
            # crops.append(cropped)
            cropimg = str(img_name.split(".")[0]) + "_cropped" + str(i) + ".jpg"
            cv2.imwrite(cropimg, cropped)
            crops.append(cropimg)

        print('Elapsed time = {}'.format(time.time() - st))
        print(all_dets)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        return crops

def vlad_query(img, k):
    import itertools
    import argparse
    import glob
    import cv2
    import pickle
    import numpy as np
    import os

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

    def query(image, k, visualDictionary, tree):
        img = cv2.imread(image)
        # img = image
        sift = cv2.xfeatures2d.SIFT_create()
        kp, descriptor = sift.detectAndCompute(img, None)

        v = VLAD(descriptor, visualDictionary)
        dist, ind = tree.query(v.reshape(1, -1), k)
        return dist, ind

    # ap = argparse.ArgumentParser()
    # ap.add_argument("-r", "--retrieve", required=True,
    #                 help="number of images to retrieve")
    # args = vars(ap.parse_args())

    # img_path = "/home/comp/e4252392/end2end"
    # k = int(args["retrieve"])

    qimg = img
    rimgIDs = []
    k = k
    pathVD = "/home/comp/e4252392/VLAD-master/discogs_hot_vd.pickle"
    treeIndex = "/home/comp/e4252392/VLAD-master/discogs_hot_tree.pickle"

    with open(treeIndex, 'rb') as f:
        indexStructure = pickle.load(f)
    with open(pathVD, 'rb') as f:
        visualDictionary = pickle.load(f)
    imageID = indexStructure[0]
    tree = indexStructure[1]
    pathImageData = indexStructure[2]
    print(pathImageData)

    # for idx, img_name in enumerate(sorted(os.listdir(img_path))):
    #     if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
    #         continue
    #     print(img_name)
    #     filepath = os.path.join(img_path, img_name)
    #     dist, ind = query(filepath, k, visualDictionary, tree)
    #     print(dist)
    #     print(ind)
    #     ind = list(itertools.chain.from_iterable(ind))
    #
    #     print(filepath)
    #     for i in ind:
    #         imageName = str(imageID[i]).split("/")[-1].split("\'")[0].split(".")[0]
    #         rimgIDs.append(imageName)
    #         print(rimgIDs[-1])

    # rimgIDs = []
    # for qimg in qimgs:
    dist, ind = query(qimg, k, visualDictionary, tree)
    print(dist)
    print(ind)
    ind = list(itertools.chain.from_iterable(ind))
    # print(filepath)
    for i in ind:
        imageName = str(imageID[i]).split("/")[-1].split("\'")[0].split(".")[0]
        rimgIDs.append(imageName)
        print(rimgIDs[-1])

    return rimgIDs



def searchdb(ids):
    import sqlite3
    import sys
    import requests
    import re
    import threading
    from queue import PriorityQueue
    from urllib.request import urlopen
    from bs4 import BeautifulSoup

    sqlite_file = '/home/comp/e4252392/test.sqlite'
    conn = sqlite3.connect(sqlite_file)
    c = conn.cursor()

    table_name = "hot4sql"
    column_name = "ImgID"
    band = "Band"
    album = "Album"
    url = "URL"
    ma = "ma4sql"
    bandname = "Bandname"

    finalinfo = "nothing"
    all_info = []

    for id in ids:
        c.execute(str('SELECT {band} FROM {tn} WHERE {cn}="'+id+'"').format(band=band, tn=table_name, cn=column_name))
        name = c.fetchall()[0][0]
        # print(name)

        c.execute(str('SELECT {album} FROM {tn} WHERE {cn}="'+id+'"').format(album=album, tn=table_name,cn=column_name))
        thealbum = c.fetchall()[0][0]
        print(thealbum)

        c.execute(str('SELECT {url} FROM {ma} WHERE {bandname} = "'+name.encode("utf-8")+'"').format(url=url, ma=ma, bandname=bandname))
        bandinfo = c.fetchall()
        # print(bandinfo)
        if len(bandinfo)<=1 and bandinfo!=[]:
            all_info.append(bandinfo[0][0])
            print("BAND: "+bandinfo[0][0])
        if len(bandinfo)>1:
            for i in range(len(bandinfo)):
                band_home = bandinfo[i][0]
                print(band_home)
                band_html = urlopen(band_home).read().decode('utf-8')
                soup = BeautifulSoup(band_html, 'html.parser')
                discog_tab = soup.find('div', id="band_tab_discography")
                links = discog_tab.find_all('a')
                for a in links:
                    if a.get_text() == "Complete discography":
                        discogs_url = a.get('href')
                        discogs_html = urlopen(discogs_url).read().decode('utf-8')
                        if thealbum in discogs_html:
                            finalinfo = band_home
                            print("FINAL: " + finalinfo)
                            all_info.append(finalinfo)
                            break
            #     if finalinfo != "nothing":
            #         break
            # if finalinfo != "nothing":
            #     break

    return all_info
    # return finalinfo



path = "/home/comp/e4252392/VLAD-master/test1118"
crops = test_frcnn(path)
for crop in crops:
    imgIDs = vlad_query(crop, 10)
    # for id in imgIDs:
    #     bandinfo = searchdb(id)
    #     print(bandinfo)
    bandinfo = searchdb(imgIDs)
    print(bandinfo)
















































