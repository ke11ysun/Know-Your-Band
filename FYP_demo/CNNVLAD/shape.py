# import argparse
# import glob
# import numpy as np
# import pickle

# def getVLADDescriptors(path, pathVD, pathCNNGT):
#     with open(pathVD, 'rb') as f:
#         visualDictionary = pickle.load(f)
#     # load cnn features
#     with open(pathCNNGT, 'rb') as f:
#         pkl = pickle.load(f)
#         objfs = pkl[1]
#         scenefs = pkl[2]
#         img_names = pkl[0]

#     # row_objfs = []
#     # col_objfs = []
#     new_scenefs = []
#     # row_scenefs = []
#     # col_scenefs = []
#     new_objfs = []
#     for scenef in scenefs:
#         # print(scenef.shape)
#         #scenef = np.reshape(scenef, 2048)
#         # row_scenefs.append(scenef.shape[1])
#         # col_scenefs.append(scenef.shape[2])
#         if scenef.shape[1]>2:
#             scenef = scenef[:,:2,:,:]
#         if scenef.shape[1]<2:
#             npad = ((0, 0), (0, 2-scenef.shape[1]), (0, 0), (0, 0))
#             scenef = np.pad(scenef, pad_width=npad, mode='constant', constant_values=0)
#         if scenef.shape[2]>2:
#             scenef = scenef[:,:,:2,:]
#         if scenef.shape[2]<2:
#             npad = ((0, 0), (0, 0), (0, 2-scenef.shape[2]), (0, 0))
#             scenef = np.pad(scenef, pad_width=npad, mode='constant', constant_values=0)
#         print(scenef.shape)  

#     for objf in objfs:
#         # print(objf.shape)
#         #objf = np.reshape(objf, 200704)
#         # row_objfs.append(objf.shape[1])        
#         # col_objfs.append(objf.shape[2])
#         if objf.shape[1]>38:
#             objf = objf[:,:38,:,:]
#         if objf.shape[1]<38:
#             npad = ((0, 0), (0, 38-objf.shape[1]), (0, 0), (0, 0))
#             objf = np.pad(objf, pad_width=npad, mode='constant', constant_values=0)
#         if objf.shape[2]>103:
#             objf = objf[:,:,:103,:]
#         if objf.shape[2]<103:
#             npad = ((0, 0), (0, 0), (0, 103-objf.shape[2]), (0, 0))
#             objf = np.pad(objf, pad_width=npad, mode='constant', constant_values=0)
#         print(objf.shape)
#     # print('scenef shape 1: ' + str(np.mean(row_scenefs)))
#     # print('scenef shape 2: ' + str(np.mean(col_scenefs)))
#     # print('objf shape 1: ' + str(np.mean(row_objfs)))
#     # print('objf shape 2: ' + str(np.mean(col_objfs)))


# ap = argparse.ArgumentParser()
# ap.add_argument("-o", "--output", required=True, help="Path to where VLAD descriptors will be stored")
# ap.add_argument("-p", "--image_path", required=True)
# ap.add_argument("--vd",required=True)
# ap.add_argument("--cnngt", default='placeholder')
# args = vars(ap.parse_args())

# output = args["output"]
# path = args["image_path"]
# pathVD = args["vd"]
# pathCNNGT = args["cnngt"]

# print("estimating VLAD descriptors using SIFT  for dataset: /" + path + " and visual dictionary: /" + pathVD)
# getVLADDescriptors(path, pathVD, pathCNNGT)





















import pickle
import pprint
# with open('cnngt_logos_pre_cpu3.pickle', 'rb') as f:
#     pprint.pprint(pickle.load(f))

# with open('sift_logos_cpu3.pickle', 'rb') as f:
#     pprint.pprint(pickle.load(f))

# with open('fc_logos_cpu3.pickle', 'rb') as f:
#     pprint.pprint(pickle.load(f)[1][38])

with open('cnngt_logos_preres.pickle', 'rb') as f:
    pprint.pprint(pickle.load(f))


# '/users/sunjingxuan/desktop/logos/127696_logo.jpg'
# array([  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#          0.00000000e+00,   4.31693329e+02,   4.40025749e+01,
#          5.79174011e+02,   0.00000000e+00,   1.65357178e+03,
#          7.27699280e+02,   2.69128387e+02,   8.72016541e+02,
#          0.00000000e+00,   8.74435425e+02,   1.80234268e+02,
#          0.00000000e+00,   0.00000000e+00,   2.25031281e+02,
#          0.00000000e+00,   0.00000000e+00,   6.28501415e+00,
#          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#          2.24060028e+02,   0.00000000e+00,   4.52697113e+02,
#          1.86202173e+03,   7.40730225e+02,   0.00000000e+00,
#          0.00000000e+00,   1.44830542e+03,   8.48327148e+02,
#          0.00000000e+00,   0.00000000e+00,   1.31730331e+02,
#          0.00000000e+00,   4.43376617e+02,   0.00000000e+00,
#          5.00734375e+02,   0.00000000e+00,   6.13102295e+02,
#          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#          0.00000000e+00,   9.02935059e+02,   0.00000000e+00,
#          2.70082031e+02,   1.64963898e+02,   0.00000000e+00,
#          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#          0.00000000e+00,   0.00000000e+00,   1.59162415e+03,
#          0.00000000e+00,   5.44284180e+02,   5.23274994e+00,
#          0.00000000e+00,   0.00000000e+00,   6.27751221e+02,
#          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#          0.00000000e+00,   0.00000000e+00,   1.42224524e+03,
#          1.50418481e+03,   0.00000000e+00,   0.00000000e+00,
#          0.00000000e+00,   1.01553955e+02,   2.08624634e+03,
#          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#          3.77550964e+01,   0.00000000e+00,   0.00000000e+00,
#          4.78020233e+02,   0.00000000e+00,   4.08600830e+02,
#          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#          0.00000000e+00,   1.54321326e+03,   2.08754364e+02,
#          7.17058167e+02,   0.00000000e+00,   0.00000000e+00,
#          2.26287048e+02,   1.09423254e+03,   0.00000000e+00,
#          0.00000000e+00,   7.14641296e+02,   1.86772797e+02,
#          5.26169357e+01,   6.49833862e+02,   1.13181604e+03,
#          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#          0.00000000e+00,   3.46268494e+02,   0.00000000e+00,
#          0.00000000e+00,   1.02669653e+03,   0.00000000e+00,
#          3.06780182e+02,   1.39621118e+03,   2.14155121e+02,
#          0.00000000e+00,   4.25890320e+02,   0.00000000e+00,
#          0.00000000e+00,   0.00000000e+00,   8.41631317e+00,
#          6.42875061e+02,   3.72454285e+02,   6.55062332e+01,
#          0.00000000e+00,   5.21236145e+02,   4.95998840e+02,
#          1.38810928e+02,   0.00000000e+00,   0.00000000e+00,
#          0.00000000e+00,   6.66294067e+02,   1.65583850e+03,
#          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#          4.93710327e+02,   9.41153748e+02,   0.00000000e+00,
#          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#          3.18680939e+02,   0.00000000e+00,   0.00000000e+00,
#          0.00000000e+00,   0.00000000e+00,   3.26951996e+02,
#          0.00000000e+00,   0.00000000e+00,   6.36516052e+02,
#          2.26184341e+02,   0.00000000e+00,   4.26564301e+02,
#          1.18201599e+03,   0.00000000e+00,   0.00000000e+00,
#          0.00000000e+00,   1.27262488e+03,   7.38691040e+02,
#          0.00000000e+00,   0.00000000e+00,   9.27165161e+02,
#          2.60161743e+02,   0.00000000e+00,   0.00000000e+00,
#          0.00000000e+00,   2.54689667e+02,   0.00000000e+00,
#          5.09571442e+02,   5.22582214e+02,   2.57867340e+02,
#          0.00000000e+00,   3.19111359e+02,   0.00000000e+00,
#          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#          0.00000000e+00,   0.00000000e+00,   6.00374878e+02,
#          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#          2.45393738e+02,   0.00000000e+00,   1.26628326e+02,
#          2.65961304e+02,   0.00000000e+00,   9.14324219e+02,
#          1.34107193e+02,   0.00000000e+00,   0.00000000e+00,
#          2.04820251e+00,   8.49417496e+00,   0.00000000e+00,
#          0.00000000e+00,   0.00000000e+00,   1.43158374e+03,
#          0.00000000e+00,   0.00000000e+00,   6.28091187e+02,
#          0.00000000e+00,   1.69935632e+03,   0.00000000e+00,
#          0.00000000e+00,   0.00000000e+00,   1.17840012e+02,
#          0.00000000e+00,   2.71725006e+02,   0.00000000e+00,
#          0.00000000e+00,   1.42476208e+03,   2.20766754e+02,
#          0.00000000e+00,   4.45850067e+02,   0.00000000e+00,
#          0.00000000e+00,   3.13517029e+02,   0.00000000e+00,
#          0.00000000e+00,   6.14507935e+02,   0.00000000e+00,
#          0.00000000e+00,   4.29649658e+02,   0.00000000e+00,
#          4.55166107e+02,   0.00000000e+00,   0.00000000e+00,
#          0.00000000e+00,   6.61820984e+01,   9.53434601e+01,
#          6.39597168e+02,   5.29144714e+02,   0.00000000e+00,
#          1.24291077e+02,   3.91705627e+02,   1.47922778e+03,
#          0.00000000e+00,   1.90836456e+02,   1.62974268e+03,
#          1.38621277e+03,   0.00000000e+00,   0.00000000e+00,
#          0.00000000e+00,   3.69308594e+02,   0.00000000e+00,
#          2.85475159e+02,   0.00000000e+00,   3.78263123e+02,
#          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#          1.00200806e+03], dtype=float32)






