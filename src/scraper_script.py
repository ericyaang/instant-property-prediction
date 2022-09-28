from scraper.scraper_olx import get_infos, get_urls_df
import pandas as pd
from datetime import datetime as dt
import json

class OlxScraper:
    def __init__(self, pages):
        self.pages = pages

    def get_urls(self):
        centro = get_urls_df('centro', self.pages)
        continente = get_urls_df('continente', self.pages)
        leste = get_urls_df('leste', self.pages)
        Norte = get_urls_df('norte', self.pages)
        sul = get_urls_df('sul', self.pages)
        return pd.concat([sul], ignore_index=True)
    
    def save_infos(self, data_path):
        infos = []
        links = self.get_urls()
        for index, row in links.iterrows():
            try:
                temp = get_infos(row[0], row[1], row[2], row[3], row[4])
                infos.append(temp)
            except:
                None
            print(f'Páginas registradas: {index}', end='\r')

        now_ = dt.now().strftime('%d%m%y-%H%M%S') 
        with open(data_path +
          now_ + '.json', 'w', encoding='utf-8') as f:
            json.dump(infos, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    n_pages= 1
    data_dir = "data/raw/olx_"
    web = OlxScraper(n_pages)
    web.save_infos(data_dir)


