import sys
import requests
import re
import threading
from queue import PriorityQueue
from urllib.request import urlopen
from bs4 import BeautifulSoup


def scrape_from_url(band_home):
    band_html = urlopen(band_home).read().decode('utf-8')
    # print(band_html)
    band_id = (band_home.split("/")[5])
    # print(band_id)
    soup = BeautifulSoup(band_html, 'html.parser')
    band_name = soup.find('h1').get_text()
    # print(band_name)
    band_stats = soup.find('div', id="band_stats")
    band_stats_dd = band_stats.find_all('dd')
    country = band_stats_dd[0].get_text()
    location = band_stats_dd[1].get_text()
    status = band_stats_dd[2].get_text()
    formed_in = band_stats_dd[3].get_text()
    genre = band_stats_dd[4].get_text()
    theme = band_stats_dd[5].get_text()
    label = band_stats_dd[6].get_text()
    years_active = band_stats_dd[7].get_text().strip()
    # print(country + "," + location + "," + status + "," + formed_in + "," + genre + "," + theme + "," + label + "," + years_active)
    current_members = []
    members_table = soup.find('div', id="band_tab_members_current")
    # print(type(members_table))
    tmembers = members_table.find_all('a', class_="bold")
    for tmember in tmembers:
        current_members.append(tmember.get_text())
    current_members_str = "^".join(map(str, current_members))
    # print(current_members)
    similar_artists = []
    similar_artists_url = soup.find('a', title="Similar artists").get('href')
    similar_artists_html = urlopen(similar_artists_url).read().decode('utf-8')
    sa_soup = BeautifulSoup(similar_artists_html, 'html.parser')
    similar_artists_trs = sa_soup.find('tbody').find_all('tr')
    for similar_artists_tr in similar_artists_trs:
        if similar_artists_tr.find('td').get('id') == "no_artists":
            similar_artists.append("No similar artists!")
        else:
            similar_artists.append(similar_artists_tr.find('td').get_text())
    similar_artists_str = "^".join(map(str, similar_artists))
    # print(similar_artists)
    related_links = []
    related_links_url = soup.find('a', title="Related links").get('href')
    related_links_html = urlopen(related_links_url).read().decode('utf-8')
    rl_soup = BeautifulSoup(related_links_html, 'html.parser')
    related_links_tags = rl_soup.find_all('a', target="_blank")
    for related_link_tag in related_links_tags:
        related_links.append(related_link_tag.get_text() + ":" + related_link_tag.get('href') + " ")
    related_links_str = "^".join(map(str, related_links))
    # print(related_links)
    read_more = soup.find('a', string="Read more")
    if read_more == None:
        comment = soup.find('div', class_="band_comment clear").get_text().strip()
    else:
        comment_url = "https://www.metal-archives.com/band/read-more/id/" + band_id
        comment_html = urlopen(comment_url).read().decode('utf-8')
        comment = BeautifulSoup(comment_html, 'html.parser').get_text()
    # print(comment)
    logo_tag = soup.find('a', id="logo")
    if logo_tag != None:
        logo_url = logo_tag.get('href')
    else:
        logo_url = "Logo not found!"
    # print(logo_url)

    text = " | ".join((band_id, band_name, country, location, status, formed_in, genre, theme, label, years_active,
                       current_members_str, similar_artists_str, related_links_str, comment, logo_url,
                       band_home)) + "\n\n"
    # text = band_name + "," + band_home + "\n"
    print(text)
    return text



path = '/users/sunjingxuan/desktop/fyp/a41.txt'
data = 'default'
inf = open(path, 'r')
#
# test_urls = ["https://www.metal-archives.com/bands/Mastodon/1361\n",
#              "https://www.metal-archives.com/bands/Abrekadaver/3540306565\n",
#              "https://www.metal-archives.com/bands/Adav%C3%A4nt/3540336315\n",
#              "https://www.metal-archives.com/bands/A_Baptism_by_Fire/3540412046\n",
#              "https://www.metal-archives.com/bands/A_Blind_Prophecy/18563\n"]

bands_home = []

# for i in range(0, len(test_urls)):
while data != "":
    data = inf.readline()
    # data = test_urls[i]
    if 'http' in data:
        band_home = data.strip("\n")
        bands_home.append(band_home)
    # print(bands_home)

outf = open("/users/sunjingxuan/desktop/fyp/info_a41.txt", "w")
for band_home in bands_home:
    text = scrape_from_url(band_home)
    outf.write(text)
outf.close()