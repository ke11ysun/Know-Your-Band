import os

# #ma
# final = []
# path = '/users/sunjingxuan/desktop/fyp/0_data/0_bands_info/'
# for filename in enumerate(sorted(os.listdir(path))):
#     print(filename)
#     bands = []
#     inf = open(path+list(filename)[1], 'r')
#     data = inf.read()
#
#     entries = data.split("\n\n")
#     count = 0
#     for i in range(len(entries)):
#         infos = entries[i].split(" | ")
#         # print(infos)
#         bandid = infos[0]
#         if len(infos) >= 2:
#             bandname = infos[1]
#         bandurl = infos[-1]
#         if bandid != bandurl.split("/")[-1]:
#             for j in range(i + 1, len(entries)):
#                 if bandid == entries[j].split(" | ")[-1].split("/")[-1]:
#                     bandurl = entries[j].split(" | ")[-1]
#                     break
#         if bandname != "":
#             bands.append(bandid + "," + bandname + "," + bandurl)
#             count = count + 1
#             print("Band" + str(count) + ": " + bands[-1] + "\n")
#
#
#     for band in bands:
#         if len(band.split(",")) == 3:
#             id = band.split(",")[0]
#             name = band.split(",")[1]
#             url = band.split(",")[2]
#             if " " not in id and "https://" in url:
#                 # final.append(name + "," + url + "\n")
#                 final.append(url.split("/")[-1] + "," + name + "," + url + "\n")
#                 print(final[-1])
#     print(len(final))
#
#
#
# # outf = open("/users/sunjingxuan/desktop/fyp/nameurl.txt", "w")
# with open('/users/sunjingxuan/desktop/fyp/test.csv', encoding='utf-8', mode='w+') as outf:
#     outf.write("ID,Name,URL\n")
#     for a in final:
#         outf.write(a)
#     outf.close()




#discogs
final = []
path = '/users/sunjingxuan/desktop/fyp/discogs_hot.txt'
inf = open(path, 'r')
data = inf.read()
entries = data.split("\n")
for entry in entries[:-1]:
    infos = entry.split(",")
    album = infos[0]
    band = infos[1]
    imgurl = infos[2]

    tokens = band.split(" ")
    if "(" in tokens[-1] and ")" in tokens[-1]:
        tokens = tokens[:-1]
        band = " ".join(tokens)
    imgid = imgurl.split("/")[-1].split(".")[0]

    e = album+","+str(band)+","+imgurl+","+imgid+"\n"
    final.append(e)
    print(final[-1])

# outf = open("/users/sunjingxuan/desktop/fyp/hot4sql.csv", "w")
with open('/users/sunjingxuan/desktop/fyp/hot4sql.csv', encoding='utf-8', mode='w+') as outf:
    outf.write("Album,Band,ImgURL,ImgID\n")
    for a in final:
        outf.write(a)
    outf.close()