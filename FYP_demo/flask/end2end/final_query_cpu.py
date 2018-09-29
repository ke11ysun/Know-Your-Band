from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
 
sys.path.append('./end2end')
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers


def format_img_size(img, C):
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
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio

def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))
    return (real_x1, real_y1, real_x2 ,real_y2)


def test_frcnn(img_name, config, model):
    from keras.models import Model
    from keras_frcnn.RoiPoolingConv import RoiPoolingConv
    import keras_frcnn.resnet as nn

    config_output_filename = config
    with open(config_output_filename, 'rb') as f_in:
        C = pickle.load(f_in)

    C.use_horizontal_flips = False
    C.use_vertical_flips = False
    C.rot_90 = False
    C.num_rois = 32
    C.model_path = model

    class_mapping = C.class_mapping
    if 'bg' not in class_mapping:
        class_mapping['bg'] = len(class_mapping)
    class_mapping = {v: k for k, v in class_mapping.items()}
    print(class_mapping)

    num_features = 1024
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

    # crop
    bbox_threshold = 0.8
    crops = []   
    # img_path = path
    # for idx, img_name in enumerate(sorted(os.listdir(img_path))):
    #     if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
    #         continue
    print(img_name)

    st = time.time()
    # filepath = os.path.join(img_path,img_name)
    # img = cv2.imread(filepath)
    img = cv2.imread(img_name)

    X, ratio = format_img(img, C)
    if K.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))

    [Y1, Y2, F] = model_rpn.predict(X)
    R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    bboxes = {}
    probs = {}
    # roifs = {}
    # scenefs = {}

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

        # out_roi_pool = RoiPoolingConv(14, C.num_rois)([feature_map_input, roi_input])
        # haha = Model(inputs=[feature_map_input, roi_input], outputs=out_roi_pool)
        # all_roifs = haha.predict([F, ROIs])
        # [all_scenefs, P_cls, P_regr] = model_classifier_only.predict([F, ROIs
        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

        for ii in range(P_cls.shape[1]):
            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []
                # roifs[cls_name] = []
                # scenefs[cls_name] = []

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
            # roifs[cls_name].append(all_roifs[0, ii, :])
            # scenefs[cls_name].append(all_scenefs[0, ii, :])

    all_dets = []
    for key in bboxes:
        bbox = np.array(bboxes[key])
        prob = np.array(probs[key])
        # roif = np.array(roifs[key])
        # scenef = np.array(scenefs[key])

        # new_boxes, new_probs, new_roifs, new_scenefs = roi_helpers.nmsf(bbox, prob, roif, scenef, overlap_thresh=0.5)
        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk,:]
            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
            all_dets.append((key,100*new_probs[jk],real_x1,real_x2,real_y1,real_y2))

    crops.append([])
    crops[len(crops) - 1].append(img_name)
    for i in range(len(all_dets)):
        cropped = img[all_dets[i][4]:all_dets[i][5], max(0, all_dets[i][2]):all_dets[i][3]]
        # croppath = '/home/comp/e4252392/end2end/crops0317'
        # croppath = '/users/sunjingxuan/pycharmprojects/end2end/crops0317_cpu'
        croppath = '/users/sunjingxuan/desktop/FYP_demo/flask/end2end/crops_demo'
        cropname = os.path.join(croppath, str(img_name.split(".")[0].split("/")[-1]) + "_cropped" + str(i) + ".jpg")
        cv2.imwrite(cropname, cropped)

        # append key, cropname
        crops[len(crops)-1].append((all_dets[i][0], cropname))

    print('Elapsed time = {}'.format(time.time() - st))
    print(all_dets)
    return crops


def vlad_query(cimg, k, visualDictionary_album, visualDictionary_logo, indexStructure_album, indexStructure_logo):
    import itertools
    import cv2
    import pickle
    import numpy as np

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
        dist = []
        ind = []
        img = cv2.imread(image)
        sift = cv2.xfeatures2d.SIFT_create()
        try:
            kp, descriptor = sift.detectAndCompute(img, None)
            v = VLAD(descriptor, visualDictionary)
            dist, ind = tree.query(v.reshape(1, -1), k)
        except cv2.error:
            print("OpeCV Error: Bad argument (image is empty or has incorrect depth! but pass.")
        except ValueError:
            print("ValueError: nan! but pass.")
        return dist, ind

    # vlad_query
    rimgIDs = []
    k = k
    imageID_album = indexStructure_album[0]
    tree_album = indexStructure_album[1]
    # pathImageData = indexStructure_album[2]
    # print(pathImageData)
    imageID_logo = indexStructure_logo[0]
    tree_logo = indexStructure_logo[1]

    print(cimg)
    # if key=album, query album tree; if key=logo, query logo tree
    if cimg[0] == "album":
        dist, ind = query(cimg[1], k, visualDictionary_album, tree_album)
    if cimg[0] == "logo":
        dist, ind = query(cimg[1], k, visualDictionary_logo, tree_logo)

    # dist, ind = query(cimg, k, visualDictionary, tree)
    if len(dist) != 0 and len(ind) != 0:
        print(dist)
        print(ind)
        ind = list(itertools.chain.from_iterable(ind))
        # print(filepath)
        for i in ind:
            if cimg[0] == "album":
                # print(str(imageID_album[i]))
                imageName = str(imageID_album[i]).split("/")[-1].split("\'")[0].split(".")[0]
            if cimg[0] == "logo":
                # print(str(imageID_logo[i]))
                # imageName = str(imageID_logo[i]).split(".")[0].split("_")[0]
                imageName = str(imageID_logo[i]).split("/")[-1]
            rimgIDs.append(imageName)
            print(rimgIDs[-1])
    return rimgIDs


def searchdb(ids, db):
    import sqlite3
    # from urllib2 import urlopen
    # import urllib2
    from urllib.request import urlopen
    from urllib.error import HTTPError, URLError
    from bs4 import BeautifulSoup

    # rebuild ma4sql, + colunm = BandID
    sqlite_file = db
    conn = sqlite3.connect(sqlite_file)
    c = conn.cursor()

    hot = "hot4sql"
    hotid = "ImgID"
    hotband = "Band"
    hotalbum = "Album"
    hotimgurl = "ImgURL"
    ma = "ma4sql"
    maname = "Name"
    maid = "ID"
    url = "URL"

    finalinfo = "nothing"
    all_info = []
    info4one = []

    # if album, hot/ma; else logo, directly ma
    for id in ids:
        try:
            if "R" in id:
                # print("entered search hot table.")
                c.execute(str('SELECT {band} FROM {hot} WHERE {id}="'+id+'"').format(band=hotband, hot=hot, id=hotid))
                name = c.fetchall()[0][0]
                print(name)

                c.execute(str('SELECT {album} FROM {hot} WHERE {id}="'+id+'"').format(album=hotalbum, hot=hot, id=hotid))
                thealbum = c.fetchall()[0][0]
                print(thealbum)

                c.execute(str('SELECT {imgURL} FROM {hot} WHERE {id}="'+id+'"').format(imgURL=hotimgurl, hot=hot, id=hotid))
                theimgURL = c.fetchall()[0][0]
                print(theimgURL)

                c.execute(str('SELECT {url} FROM {ma} WHERE {bandname} = "'+name+'"').format(url=url, ma=ma, bandname=maname))
                bandinfo = c.fetchall()
                # print(bandinfo)
                if bandinfo == []:
                    not_found = "Sorry, band of interest is not included in Metal Archive database."
                    all_info.append((name, thealbum, id, not_found))
                    print("NOT FOUND: " + not_found)
                if len(bandinfo)<=1 and bandinfo!=[]:
                    all_info.append((name, thealbum, id, bandinfo[0][0]))
                    info4one.append({'name':name, 'album': thealbum, 'imgURL': theimgURL, 'bandURL': bandinfo[0][0]})
                    print("BAND: "+bandinfo[0][0])
                if len(bandinfo)>1:
                    breaker = False
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
                                    all_info.append((name, thealbum, id, finalinfo))
                                    info4one.append({'name':name, 'album': thealbum, 'imgURL': theimgURL, 'bandURL': finalinfo})
                                    breaker = True
                                break
                        if breaker:
                            break
            else:
                # print("entered search ma table.")
                logoName_suffix = id
                # print(id)
                id = str(id.split("_")[0])
                # print(id)
                c.execute(str('SELECT {bandname} FROM {ma} WHERE {bandID}="' + id + '"').format(bandname=maname, ma=ma, bandID=maid))
                name = c.fetchall()[0][0]
                print(name)

                c.execute(str('SELECT {url} FROM {ma} WHERE {bandID} = "' + id + '"').format(url=url, ma=ma, bandID=maid))
                bandinfo = c.fetchall()[0][0]
                print("LOGO: " + bandinfo)
                all_info.append((name, id, bandinfo))

                one = id[0]+"/"
                two = three = four = ""
                if len(id)>1:
                    two = id[1]+"/"
                    if len(id)>2:
                        three = id[2]+"/"
                        if len(id)>3:
                            four = id[3]+"/"
                theimgURL = 'https://www.metal-archives.com/images/' + one + two + three + four + logoName_suffix
                info4one.append({'name':name, 'imgURL': theimgURL, 'bandURL': bandinfo})
                
        except IndexError:
            print("IndexError: list index out of range! but pass.")
        except HTTPError:
            print("urllib2.HTTPError: HTTP Error 500: Internal Server Error! but pass.")
        except URLError:
            print("urllib2.URLError: 503 Service Unavailable! but pass.")
        
    # return all_info
    return info4one





def final_query_cpu(test_img):
    import json
    # parser = OptionParser()
    # parser.add_option("--test", dest="path_test", default="/users/sunjingxuan/desktop/test_output/test1118/bufftest")
    # parser.add_option("--albumvd", dest="path_albumvd", default="/users/sunjingxuan/pycharmprojects/VLAD/vd_discogshot_cpu3.pickle")
    # parser.add_option("--logovd", dest="path_logovd", default="/users/sunjingxuan/pycharmprojects/VLAD/vd_logos_cpu3.pickle")
    # parser.add_option("--db", dest="path_db", default="/users/sunjingxuan/desktop/FYP/smalldb0124.sqlite")
    # # might change, current model=.78
    # parser.add_option("--config", dest="path_config", default="/users/sunjingxuan/desktop/retrain_models/retrain_config_init.pickle")
    # parser.add_option("--model", dest="path_model", default="/users/sunjingxuan/desktop/retrain_models/retrain_model_init.hdf5")
    # # change
    # parser.add_option("--albumtree", dest="path_albumtree", default="/users/sunjingxuan/pycharmprojects/VLAD/tree_vlad_hot_cpu3.pickle")
    # parser.add_option("--logotree", dest="path_logotree", default="/users/sunjingxuan/pycharmprojects/VLAD/tree_vlad_logos_cpu3.pickle")

    # (options, args) = parser.parse_args()
    path = test_img
    path_config = "/users/sunjingxuan/desktop/retrain_models/retrain_config_init.pickle"
    path_model = "/users/sunjingxuan/desktop/retrain_models/retrain_model_init.hdf5"
    path_albumvd = "/users/sunjingxuan/pycharmprojects/VLAD/vd_discogshot_cpu3.pickle"
    path_logovd = "/users/sunjingxuan/pycharmprojects/VLAD/vd_logos_cpu3.pickle"
    path_albumtree = "/users/sunjingxuan/pycharmprojects/VLAD/tree_vlad_hot_cpu3.pickle"
    path_logotree = "/users/sunjingxuan/pycharmprojects/VLAD/tree_vlad_logos_cpu3.pickle"
    path_db = "/users/sunjingxuan/desktop/finaldb.sqlite"

    crops = test_frcnn(path, path_config, path_model)
    # print(len(crops))
    # load 2 vd/trees, one for album, one for logo
    with open(path_albumtree, 'rb') as f:
        indexStructure_album = pickle.load(f)
    with open(path_albumvd, 'rb') as f:
        visualDictionary_album = pickle.load(f)

    with open(path_logotree, 'rb') as f:
        indexStructure_logo = pickle.load(f)
    with open(path_logovd, 'rb') as f:
        visualDictionary_logo = pickle.load(f)


    json4client = []
    # each qimg
    for crop in crops:
        info4dump = {}
        info4dump['info'] = []
        album_list = []
        logo_list = []

        print("qimg: " + crop[0])
        print("len(crop): " + str(len(crop)))
        # each crop of the qimg
        for i in range(1, len(crop)):
            imgIDs = vlad_query(crop[i], 10, visualDictionary_album, visualDictionary_logo, indexStructure_album, indexStructure_logo)
            bandinfo = "Bad Crop!"
            if len(imgIDs) != 0:
                bandinfo = searchdb(imgIDs, path_db)
            print(bandinfo)

            # format json
            if type(bandinfo) is not str:
                for record in bandinfo:
                    if len(record) is 4:
                        if record['album'] not in album_list:
                            album_list.append(record['album'])
                            info4dump['info'].append(record)
                    elif len(record) is 3:
                        if record['name'] not in logo_list:
                            logo_list.append(record['name'])
                            info4dump['info'].append(record)

                    # info4dump['info'].append(record)

        jsonfolder = '/users/sunjingxuan/desktop/FYP_demo/flask/end2end/jsons_demo'
        # jsonpath = os.path.join(jsonfolder, 'jsondump_' + crop[0].split("/")[-1] + '.json')
        jsonpath = 'jsondump_' + crop[0].split("/")[-1] + '.json'
        with open(os.path.join(jsonfolder, jsonpath), 'w') as outfile:  
            json.dump(info4dump, outfile, indent=4)
        print('Finished dumping \m/')
        # json4client.append(jsonpath)

    # return json4client
    return jsonfolder, jsonpath
