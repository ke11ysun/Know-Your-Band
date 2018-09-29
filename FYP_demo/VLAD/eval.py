# import sys
# from PyQt4.QtGui import *
# from PyQt4.QtCore import *
# from PyQt4.QtWebKit import *
# from lxml import html
#
# class Render(QWebPage):
#   def __init__(self, url):
#     self.app = QApplication(sys.argv)
#     QWebPage.__init__(self)
#     self.loadFinished.connect(self._loadFinished)
#     self.mainFrame().load(QUrl(url))
#     self.app.exec_()
#
#   def _loadFinished(self, result):
#     self.frame = self.mainFrame()
#     self.app.quit()
#
# # url = 'http://pycoders.com/archive/'
# url = 'https://www.metal-archives.com/lists/A'
# #This does the magic.Loads everything
# r = Render(url)
# #result is a QString.
# result = r.frame.toHtml()
#
# #QString should be converted to string before processed by lxml
# formatted_result = str(result.toAscii())
#
# #Next build lxml tree from formatted_result
# tree = html.fromstring(formatted_result)
#
# #Now using correct Xpath we are fetching URL of archives
# # archive_links = tree.xpath('//div[@class="campaign"]/a/@href')
# # print archive_links
# band_names = tree.xpath('//td[@class=" sorting_1"]/a/text()')
# # //*[@id="bandListAlpha"]/tbody/tr[1]/td[1]/a
# # //*[@id="bandListAlpha"]/tbody/tr[2]/td[1]/a
# # band_names = tree.xpath('//h1[@class="page_title"]/text()')
# print (band_names)
#
# # http://www.metal-archives.com/browse/ajax-letter/l/A/json/1?sEcho=1&iColumns=4&sColumns=&iDisplayStart=0&iDisplayLength=500&mDataProp_0=0&mDataProp_1=1&mDataProp_2=2&mDataProp_3=3&iSortCol_0=0&sSortDir_0=asc&iSortingCols=1&bSortable_0=true&bSortable_1=true&bSortable_2=true&bSortable_3=false&_=1482634713018
# # 8de0ec10-c438-453e-994a-09da49c709ba




# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#
# datagen = ImageDataGenerator(
#         rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest')
#
# img = load_img('/users/sunjingxuan/desktop/fyp/0_reallife/p0.jpg')  # this is a PIL image
# x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
# x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
#
# # the .flow() command below generates batches of randomly transformed images
# # and saves the results to the `preview/` directory
# i = 0
# for batch in datagen.flow(x, batch_size=1,
#                           save_to_dir='/users/sunjingxuan/desktop/fyp/0_reallife/preview', save_prefix='p0', save_format='jpeg'):
#     i += 1
#     if i > 20:
#         break  # otherwise the generator would loop indefinitely










# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread('/users/sunjingxuan/desktop/fyp/0_repo/mastoeos_crop.jpg')
# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# sift = cv2.xfeatures2d.SIFT_create()
# kp = sift.detect(img,None)
# (kps, descs) = sift.detectAndCompute(img, None)
# print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
# print(descs[0].shape)
# np.save('purple.npy', descs)
# print(type(np.load('purple.npy')))
# # print(kps)
# # print(descs)
# # array0 = np.load('purple.npy')[0]
# # array1 = np.load('purple.npy')[1]
# # final_array = np.concatenate(([array0], [array1]), axis=1)
# # print(final_array.shape)
# # print(type(final_array))
# cv2.drawKeypoints(img,kp,img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imwrite('sift_keypoints.jpg',img)
#
# # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
# # hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
# # print(hist.shape)
# # cv2.imshow('image', hist)
# # cv2.waitKey(0)




















#
# # obj = np.load('new_roifs.npy')
# # scene = np.load('new_scenefs.npy')
# from sklearn.decomposition import PCA
# pca = PCA(n_components=25, svd_solver='full')
# scene = np.ones((512,128)).reshape(256,256)
# # scene = np.load('/users/sunjingxuan/pycharmprojects/keras-frcnn-master/new_scenefs.npy')
# # print(scene.shape)
# # scene = scene.reshape(-1, 128)
# # print(scene.shape)
# print(pca.fit_transform(scene).shape)


# #
# # X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# # print(X.shape)
# # pca = PCA(n_components=1)
# # print(pca.fit_transform(X).shape)


# X = np.array([1,2,35,4,58,6,72,8])
# Y = np.array([2,20,3,49,5,63,7,8])
# Z = np.vstack((X,Y))
# print(Z)
#
# from sklearn.decomposition import PCA
# pca = PCA(n_components=7)
# print(Z)
# pca.fit(Z)
# z = pca.transform(Z)
# print(z)
# print(pca.n_components)


#
# import numpy as np
# a = np.nan
# if np.isnan(a):
#     print(a)











# import sqlite3
# import sys
# import requests
# import re
# import threading
# from queue import PriorityQueue
# from urllib.request import urlopen
# from bs4 import BeautifulSoup
#
# sqlite_file = '/users/sunjingxuan/desktop/fyp/test.sqlite'
# conn = sqlite3.connect(sqlite_file)
# c = conn.cursor()
#
# table_name = "hot4sql"
# column_name = "ImgID"
# band = "Band"
# album = "Album"
# url = "URL"
# ma = "ma4sql"
# bandname = "Bandname"
#
# #leprous as example for filter same name bands
# c.execute('SELECT {band} FROM {tn} WHERE {cn}="R-8303225-1504119620-7835"'.format(band=band, tn=table_name, cn=column_name))
# name = c.fetchall()[0][0]
# print(name)
#
# c.execute('SELECT {album} FROM {tn} WHERE {cn}="R-8303225-1504119620-7835"'.format(album=album, tn=table_name, cn=column_name))
# thealbum = c.fetchall()[0][0]
# print(thealbum)
#
# c.execute(str('SELECT {url} FROM {ma} WHERE {bandname} = "'+name+'"').format(url=url, ma=ma, bandname=bandname))
# bandinfo = c.fetchall()
# # print(bandinfo)
#
# # c.execute('SELECT {url} FROM {ma} WHERE {bandname} = {SELECT {band} FROM {tn} WHERE {cn} = \'R-11103635-1509907428-6480\'}'.format(url=url, ma=ma, bandname=bandname, band=band, tn=table_name, cn=column_name))
#
#
# for i in range(len(bandinfo)):
#     band_home = bandinfo[i][0]
#     print(band_home)
#     band_html = urlopen(band_home).read().decode('utf-8')
#     soup = BeautifulSoup(band_html, 'html.parser')
#     discog_tab = soup.find('div', id="band_tab_discography")
#     links = discog_tab.find_all('a')
#     for a in links:
#         if a.get_text() == "Complete discography":
#             discogs_url = a.get('href')
#             discogs_html = urlopen(discogs_url).read().decode('utf-8')
#             if thealbum in discogs_html:
#                 finalinfo = band_home
#                 break
#
# print("FINAL: " + finalinfo)
#
#
# # name = "Кипелов".encode("utf-8", "ignore")
# # print(name.decode("utf-8"))
# # print(name.decode("ascii", "replace"))









# Create a 2d list.
# elements = []
#
# elements.append([])
# elements[len(elements)-1].append(1)
# elements[len(elements)-1].append(2)
# elements.append([])
# elements[len(elements)-1].append(3)
# elements[len(elements)-1].append(4)
#
# print(elements)
#
# for element in elements:
#     print("haha")
#     for i in range(0, len(element)):
#         print(element[i])







################################################################################
import cv2
import numpy as np
from matplotlib import pyplot as plt

imgnames = []
imgs = []
logopath = "/users/sunjingxuan/desktop/logos/"
albumpath = "/users/sunjingxuan/desktop/discogs_hot/"
with open("/users/sunjingxuan/desktop/aaa.txt", 'rb') as f:
    content = f.read()
lines = content.splitlines()
for line in lines:
    # print(str(line).split("\'")[1])
    if "R-" in str(line) and "www" not in str(line):
        imgname = albumpath + str(line).split("/")[-1].split("\'")[0]
        print(imgname)
        imgnames.append(imgname)
    if "logos" in str(line) and "www" not in str(line):
        imgname = logopath + str(line).split("/")[-1].split("\'")[0]
        # imgname = imgpath + str(line).split("\'")[1]
        print(imgname)
        imgnames.append(imgname)



for imgname in imgnames:
    img = cv2.imread(imgname)
    imgs.append(img)
    cv2.imshow('image', img)
    cv2.waitKey(0)
################################################################################
# with open("/users/sunjingxuan/desktop/untitled.txt", 'rb') as f:
#     content = f.read()
# lines = content.splitlines()
# for line in lines:
#     if 





