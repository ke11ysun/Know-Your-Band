from icrawler.builtin import GoogleImageCrawler

google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=4,
                                    storage={'root_dir': '/users/sunjingxuan/desktop/black_poster'})
google_crawler.crawl(keyword='black metal posters logo', max_num=1000,
                     date_min=None, date_max=None,
                     min_size=(200,200), max_size=None)