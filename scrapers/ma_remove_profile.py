import os
path = "/users/sunjingxuan/downloads/logoscrape_s_end"

directory = os.fsencode(path)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)
    if "photo" in filename or filename.endswith(".gif"):
        os.remove("/users/sunjingxuan/downloads/logoscrape_s_end/" + filename)
        print(filename + " Removed!")
        continue
    else:
        continue