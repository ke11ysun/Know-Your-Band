from icrawler.builtin import GoogleImageCrawler
from datetime import date

google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=4,
                                    storage={'root_dir': '/users/sunjingxuan/desktop/retrain_prog'})
google_crawler.crawl(keyword='prog magazine', max_num=1000,
                     date_min=None, date_max=None,
                     min_size=(200,200), max_size=None)