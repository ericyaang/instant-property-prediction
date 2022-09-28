from bs4 import BeautifulSoup
import requests
import pandas as pd
from datetime import datetime as dt
from IPython.display import clear_output

##
# get links for each ad
##

def get_urls(link, pages):
    """
    coleta o link de cada anúncio e retorna um dataframe.
    """
    links = {}
    errors = []

    for page in range(1, pages + 1):
        clear_output(wait=True)

    # request
        olxRequest = requests.get('{}?o={}'.format(link, page), headers={
                                  'User-Agent': 'Mozilla/5.0'})
        try:
            assert olxRequest.status_code == 200, 'Status code error {}'.format(
                olxRequest.status_code)
        except:
            print('Status code error {} in page {}'.format(
                olxRequest.status_code, page))
            errors.append(page)
            continue

    # soup
        soup = BeautifulSoup(olxRequest.content, 'html.parser')
        itens = soup.find('ul', {'id': 'ad-list'})
        lista = itens.find_all('li')

    # urls
        urls = []
        for i, i in enumerate(lista):
            try:
                temp_href = i.find('a').get('href')
                temp_item = i.find('a').get('data-lurker_list_id')
                temp_title = i.find('a').get('title')

        # print(temp_item)
                urls.append([temp_item, temp_title, temp_href])
            except:
                None

        links[page] = urls
        print(f'Página Atual: {page}')
        print(f'Páginas com erro: {errors}')
    return links


def get_infos(codigo, descricao, link, page, region):
    """
    Acessa cada anúncio e coleta as informações.

    """
    # request
    try:
        houseRequest = requests.get(
            link, headers={'User-Agent': 'Mozilla/5.0'})
        assert houseRequest.status_code == 200, 'Status code error'
    except:
        time.sleep(4)
        houseRequest = requests.get(
            link, headers={'User-Agent': 'Mozilla/5.0'})
        assert houseRequest.status_code == 200, 'Status code error'

    # soup
    soup = BeautifulSoup(houseRequest.content, 'html.parser')
    infos = {}
    infos['codigo'] = codigo
    infos['descricao'] = descricao
    infos['link'] = link
    infos['page'] = page
    infos['regiao'] = region

    # value
    value = soup.find(
        'div', {'data-testid': 'ad-price-wrapper'}).find('h2').get_text()
    infos['Valor'] = value

    # time
    time = soup.find(
        'div', {'class': 'sc-hmzhuo fpFNoN sc-jTzLTM iwtnNi'}).find('span').get_text()
    infos['Data'] = time

    # details
    for i in soup.find('div', {'data-testid': 'ad-properties'}):
        detailType = i.find('dt').get_text()
        try:
            value = i.find('a').get_text()
        except:
            value = i.find('dd').get_text()
        infos[detailType] = value

    # localization
    for i in soup.find_all('div', {'data-testid': 'ad-properties'})[1]:
        detailType = i.find('dt').get_text()
        try:
            value = i.find('a').get_text()
        except:
            value = i.find('dd').get_text()
        infos[detailType] = value

    return infos

##
# helper function for get_urls
##

def get_urls_df(region, pages):
    """
    função auxiliar para iterar a coleta para cada uma das regiões requeridas.

    """
    pages = get_urls(
        f'https://sc.olx.com.br/florianopolis-e-regiao/{region}/imoveis/venda', pages)

    links = pd.DataFrame([])

    for page in pages:

        temp = pd.DataFrame(pages.get(page), columns=[
                            'id', 'description', 'link'])
        temp['page'] = page
        temp['region'] = region
        links = links.append(temp, ignore_index=True)
    links = links.drop_duplicates(subset=['id'])
    return links