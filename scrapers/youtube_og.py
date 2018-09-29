# scraper for youtube watch later thumbnail
# manually collected and verified quality of 700 video thumbnails
# as youtube script only load the first 100 video into html, can only delete 1-100 to get 101-200 diplayed
# so saved the rendered html as in_paths
# extracted og urls to out_paths

import re

in_paths = ['/users/sunjingxuan/desktop/fyp/youtube_html_100.txt',
            '/users/sunjingxuan/desktop/fyp/youtube_html_200.txt',
            '/users/sunjingxuan/desktop/fyp/youtube_html_300.txt',
            '/users/sunjingxuan/desktop/fyp/youtube_html_400.txt',
            '/users/sunjingxuan/desktop/fyp/youtube_html_500.txt',
            '/users/sunjingxuan/desktop/fyp/youtube_html_600.txt',
            '/users/sunjingxuan/desktop/fyp/youtube_html_700.txt']
out_paths = ['/users/sunjingxuan/desktop/fyp/youtube_og_100.txt',
                '/users/sunjingxuan/desktop/fyp/youtube_og_200.txt',
                '/users/sunjingxuan/desktop/fyp/youtube_og_300.txt',
                '/users/sunjingxuan/desktop/fyp/youtube_og_400.txt',
                '/users/sunjingxuan/desktop/fyp/youtube_og_500.txt',
                '/users/sunjingxuan/desktop/fyp/youtube_og_600.txt',
                '/users/sunjingxuan/desktop/fyp/youtube_og_700.txt']

for i in range(0, len(in_paths)):
    inf = open(in_paths[i], 'r')
    content = inf.read()
    temp = []
    temp.extend(re.findall('"videoId":"(.*?)","playlistId"', content))

    print("Writing to path " + str(i+1) + "...")
    outf = open(out_paths[i], "w")
    for j in range(0, len(temp)):
        og_id = temp[j].split("\"")[0]
        og_url = "https://i.ytimg.com/vi/" + og_id + "/hqdefault.jpg"
        outf.write(og_url + "\n")
        # print(og_url)
    outf.close()
    print("Writen to path " + str(i+1) + ".")



# path = '/users/sunjingxuan/desktop/fyp/1013_real_youtube_og.txt'
# inf = open(path, 'r')
#
# data = inf.read()
# # print(data)
# print("Start with formatting...")
# mids = data.split("},{\"url\":\"")
#
# results = []
# for mid in mids:
#     results.append(mid.split("\"", 1)[0])
#     # print (result)
#
# outf = open("/users/sunjingxuan/desktop/fyp/1013_real_youtube_og_strip.txt", "w")
# for result in results:
#     outf.write(result)
#     if "\n" not in result:
#         outf.write("\n")
# print("Done with formatting.")
# outf.close()




















