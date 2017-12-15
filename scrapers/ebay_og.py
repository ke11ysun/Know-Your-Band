import requests
import re
from bs4 import BeautifulSoup
from urllib.request import urlopen
import time

# scrape 1000*1000 fine quality pic from ebay
# search keyword: "pre owned metal vinyl" (for the first "test" string)
# search keyword: "pre owned metal cd" (for the second "test" string)

test = "https://www.ebay.com/sch/Music/11233/i.html?_from=R40&_nkw=pre+owned+metal+vinyl&_ipg=200&rt=nc&_pgn="
# test = "https://www.ebay.com/sch/Music/11233/i.html?_from=R40&_nkw=pre+owned+metal+cd&_ipg=200&rt=nc&_pgn="
page = 49

itm_ids = []
og_urls = []

for i in range(15, page):
    page_url = test + str(i+1)
    print("Scraping page" + str(i+1) + "...")
    print(page_url)
    html = urlopen(page_url).read().decode('utf-8')
    soup = BeautifulSoup(html, 'html.parser')
    itm_tags = soup.find_all('h3', class_="lvtitle")

    for itm_tag in itm_tags:
        itm_url = itm_tag.find('a').get('href')

        # for "vinyl", use "?hash"
        # for "cd", use "?hash" and "?epid"
        itm_id_chars = re.findall('-/(.*?)\?hash=', itm_url)

        itm_id = ''
        for itm_id_char in itm_id_chars:
            itm_id = itm_id + itm_id_char
        # print(itm_id)
        if itm_id is "":
            pass
        else:
            itm_ids.append(itm_id)
    # print(itm_ids)
    print(len(itm_ids))
    print("Finish page" + str(i+1) + ".")


    # will write itm_ids to multiple files many times
    # but in case of http error, just keep the final files before crash and delete afore ones
    # naming rule: for "vinyl", "ebay_og_vn" + page number.txt
    #              for "cd" (?hash), "ebay_og_cd" + page number.txt
    #              for "cd" (?epid), "ebay_og_ep" + page number.txt
    #              uniquelines: uvn/ucd/uep
    out_path = "/users/sunjingxuan/desktop/fyp/ebay_og_vn" + str(i+1) + ".txt"
    unique_out_path = "/users/sunjingxuan/desktop/fyp/ebay_og_uvn" + str(i+1) + ".txt"
    outf = open(out_path, "w")
    for itm_id in itm_ids:
        og_url = "http://i.ebayimg.com/images/i/" + itm_id + "-0-1/s-l1000.jpg"
        outf.write(og_url + "\n")
    outf.close()

    uniqlines = set(open(out_path).readlines())
    out = open(unique_out_path, 'w').writelines(uniqlines)
    print("Writen to " + unique_out_path)

# for i in range(0, page):
#     page_url = test + str(i+1)
#     print("Scraping page" + str(i+1) + "...")
#     print(page_url)
#
#     time.sleep(3)
#     content = requests.get(page_url).text
#     itm_ids.extend(re.findall('-/(.*?)\?hash=', content))
#     # print(itm_ids)
#     print("Finish page" + str(i+1) + ".")
#
# content = requests.get(test).text
# itm_ids.extend(re.findall('-/(.*?)\?hash=', content))
# print(itm_ids)
#
# outf = open(out_path, "w")
# for itm_id in itm_ids:
#     og_url = "http://i.ebayimg.com/images/i/" + itm_id + "-0-1/s-l1000.jpg"
#     outf.write(og_url + "\n")
# outf.close()
# # print(og_urls)
#
# uniqlines = set(open(out_path).readlines())
# out = open(unique_out_path, 'w').writelines(uniqlines)

# https://i.ebayimg.com/thumbs/images/g/5XsAAOSwT6JZ2jfZ/s-l225.jpg
# http://i.ebayimg.com/images/i/122745178179-0-1/s-l1000.jpg
# http://www.ebay.com/itm/Lot-of-5-Metal-LP-Vinyl-Records-12-Black-Sabbath-Motley-Crue-Led-Zeppelin-II-/122745178179?hash=item1c942ec843:g:5XsAAOSwT6JZ2jfZ
#
# http://i.ebayimg.com/images/i/361243012541-0-1/s-l1000.jpg
# <meta Property="og:image" Content="http://i.ebayimg.com/images/i/361243012541-0-1/s-l1000.jpg" />
#
# h3 class="lvtitle"



