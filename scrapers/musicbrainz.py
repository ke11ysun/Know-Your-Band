import musicbrainzngs as mb


data = "default"
artist_names = []
inf = open("/users/sunjingxuan/desktop/fyp/top_artists_150.txt")
while data is not "":
    data = inf.readline()
    artist_names.append(data.strip())

image_list = mb.get_image_list("03367cb8-1001-451b-b06f-1da05d44da49")
print(image_list)

# mb.get_image("03367cb8-1001-451b-b06f-1da05d44da49", coverid)



