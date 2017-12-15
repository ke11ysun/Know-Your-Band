# from bs4 import BeautifulSoup
# import os
# import pprint as pp
#
# dirs = ['/users/sunjingxuan/desktop/poster/annotations',
#         '/users/sunjingxuan/desktop/reallife/annotations',
#         '/users/sunjingxuan/desktop/youtubeog/annotations']
# # counted screen1-38 in reallife annotations
# # nums_files = [438, 267, 607]
# all_files = []
# objs_data = []
# total_files = 0
# total_objs = 0
#
# for dir in dirs:
#     en_dir = os.fsencode(dir)
#     # print(type(en_dir))
#     # < class 'bytes'>
#     for file in os.listdir(en_dir):
#         filename = os.fsdecode(file)
#         # print(filename)
#         all_files.append(dir + "/" + filename)
# # print(all_files)
#
# for file in all_files:
#     with open(file) as fp:
#         # print(fp.name.split('/')[-1] )
#         total_files = total_files + 1
#         soup = BeautifulSoup(fp, 'lxml')
#         # inf = open(file, 'r')
#         # content = inf.read()
#         objs = soup.find_all('object')
#         for obj in objs:
#             label = obj.find('name').get_text()
#             # print(label)
#             xmin = obj.find('xmin').get_text()
#             ymin = obj.find('ymin').get_text()
#             xmax = obj.find('xmax').get_text()
#             ymax = obj.find('ymax').get_text()
#             img_name = '/home/comp/e4252392/try_train_frcnn/' + fp.name.split('/')[-1].split('.')[0] + '.jpg'
#             obj_data = img_name + ',' + xmin + ',' + ymin + ',' + xmax + ',' + ymax + ',' + label + '\n'
#             objs_data.append(obj_data)
#             total_objs = total_objs +1
#
# pp.pprint(objs_data)
# print('total_files: ' + str(total_files))
# print('total_objs: ' + str(total_objs))
# outf = open('/users/sunjingxuan/desktop/realdata4frcnn.txt', 'w')
# for obj_data in objs_data:
#     outf.write(obj_data)
# outf.close()
# print('Writen to file.')

# only img_name, not path yet!!! '/home/comp/e4252392/try_train_frcnn/' +

# import numpy as np
# losses = np.zeros((5, 5))
# print(type(losses))
# best_loss = np.Inf
# print(type(best_loss))
#
import pickle
# p = pickle.load( open( "config.pickle", "rb" ) )
# print(p)

objects = []
with (open("/users/sunjingxuan/desktop/1101vlad.pickle", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
print(objects[0])

# print(np.load('/users/sunjingxuan/desktop/best_loss_init.npy'))


