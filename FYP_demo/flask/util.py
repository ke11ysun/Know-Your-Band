# from urllib2 import urlopen
# import urllib2
# from bs4 import BeautifulSoup

# band_home = 
# band_html = urlopen(band_home).read().decode('utf-8')
# soup = BeautifulSoup(band_html, 'html.parser')

# thumbnail_center = soup.find('span', class_='thumbnail_center')
# imgURL = thumbnail_center.find('img').get('src')
# print(imgURL)

# import json

# data = {}  
# data['people'] = []  
# data['people'].append({  
#     'name': 'Scott',
#     'website': 'stackabuse.com',
#     'from': 'Nebraska'
# })
# data['people'].append({  
#     'name': 'Larry',
#     'website': 'google.com',
#     'from': 'Michigan'
# })
# data['people'].append({  
#     'name': 'Tim',
#     'website': 'apple.com',
#     'from': 'Alabama'
# })

# with open('data.json', 'w') as outfile:  
#     json.dump(data, outfile, indent=4)



# import pickle

# # with open(treeIndex_album, 'rb') as f:
# #     indexStructure_album = pickle.load(f, encoding='latin1')

# # imageID_album = indexStructure_album[0]
# # tree_album = indexStructure_album[1]

# # print(imageID_album[0])

# pathVD_album = "/users/sunjingxuan/pycharmprojects/VLAD/hot_vd_cpu.pickle"
# with open(pathVD_album, 'rb') as f:
#     visualDictionary_album = pickle.load(f, encoding='latin1')
#     print(visualDictionary_album)


# original = "Mary's Blood"
# encoded = str(original.encode('utf-8'))
# print(type(encoded))
# print(encoded.split("\"")[1])

