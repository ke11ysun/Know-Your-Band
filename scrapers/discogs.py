from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests
import re
from urllib.error import HTTPError
import time

# scraper for discogs, heavy metal master, by country
# page = num of master/250 per page
# countries and pages are pseudo, actually manually type in country and page



# countries = ["US", "Germany", "UK", "Europe", "Japan", "Italy", "Russia", "Spain", "Finland", "France", "Sweden"]
# pages = ["18", "14", "7", "6", "5", "3", "3", "2", "2", "2", "2"]
# for i in range(0, len(countries)):
    # default = "https://www.discogs.com/search/?limit=250&sort=title%2Casc&style_exact=Heavy+Metal&country_exact=" + country + "&page=" + page
# default = "https://www.discogs.com/search/?limit=250&sort=title%2Casc&style_exact=Heavy+Metal&type=master&country_exact=Sweden&page="
default = "https://www.discogs.com/search/?limit=250&sort=hot%2Cdesc&style_exact=Heavy+Metal&type=master&page="
page = 2
# page = 1
page_urls = []
for i in range(0, page):
    page_urls.append(default + str(i+1))
    # print(page_urls)


thumb_urls = []
albums = []
artists = []

def scrape(page_url):
    page_html = urlopen(page_url).read().decode('utf-8')
    soup = BeautifulSoup(page_html, 'html.parser')
    content = requests.get(page_url).text

    thumb_urls.extend(re.findall('img data-src="(.*?)"', content))
    # print(len(thumb_urls))
    album_tags = soup.find_all('h4')
    for album_tag in album_tags:
        albums.append(album_tag.find('a').get('title'))
        # artists.append(album_tag.find('a').get('href').split("-")[0])
    # print(albums[i * 250])
    # print(len(albums))
    # print(len(artists))


    # some albums are colaborated between more than one artists
    # num of artist_taggroups = num of albums
    # num of artist_tags in each artist_taggroups = num of colaborate artists
    # some h5 is referring to html structure
    artist_taggroups = soup.find_all('h5')
    for artist_taggroup in artist_taggroups:
        artist_tags = artist_taggroup.find_all('span', itemprop="name")
        # print(len(artist_tags))
        if len(artist_tags) is not 0:
            co_artists = artist_tags[0].get('title')
            if len(artist_tags) > 1:
                for i in range(1, len(artist_tags)):
                    co_artists = co_artists + "/" + artist_tags[i].get('title')
            artists.append(co_artists)
        else:
            pass
    # print(len(artists))


for i in range(0, len(page_urls)):
    try:
        print("Start scraping page " + str(i + 1) + "...")
        scrape(page_urls[i])
        print(len(thumb_urls))
        print(albums[i * 250])
        print(len(albums))
        print(len(artists))
        print("Finish scraping page " + str(i + 1) + ".")

    except HTTPError as err:
        print("...sleep...")
        time.sleep(60)
        print("Start scraping page " + str(i+1) + "...")
        scrape(page_urls[i])
        print(len(thumb_urls))
        print(albums[i * 250])
        print(len(albums))
        print(len(artists))
        print("Finish scraping page " + str(i + 1) + ".")

outf = open('/users/sunjingxuan/desktop/fyp/discogs_hot.txt','w')
for i in range(0, len(albums)):
    record = albums[i] + ","  + artists[i] + "," + thumb_urls[i] + "\n"
    # print(record)
    outf.write(record)
print("Writen to file.")
outf.close()