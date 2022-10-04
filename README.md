# Instant Property Value Prediction
A fast property valuation tool that uses geographical characteristics to provide a precise listing price range. 

## Table of Contents
- [Instant Property Value Prediction](#instant-property-value-prediction)
  - [Table of Contents](#table-of-contents)
  - [App](#app)
  - [How accurate is it?](#how-accurate-is-it)
  - [Data](#data)
    - [Example of raw datapoint from listings](#example-of-raw-datapoint-from-listings)
    - [Example of the datapoint from the final dataset](#example-of-the-datapoint-from-the-final-dataset)
    - [Scraper](#scraper)
  - [Setup](#setup)
  - [Limitations and improvements](#limitations-and-improvements)
  - [License](#license)
## App
``
streamlit src/run app.py 
``

## How accurate is it?
Light Gradient Boosting Machine (LightGBM) was used to estimate the lower (25%), middle (50%), and upper (75%) ranges by using a quantile loss function.

The model was trained with almost 40.000 listings from April to February in 2022 and is on average over 86% accurate.

The location was restricted to Florianópolis, the capital of the state of Santa Catarina, in the South region of Brazil. And only houses and apartments were considered. Duplicated listings or unrealistic selling prices per square meter were excluded. For more details see `config\params.json`

Results on the test data:

| Model | MAE | MAPE | R2 |
| --- | ---| ---| ---|
| Lower bound (0.25) | R$ 218,260.75 | 13.90% | 85.84% |
| Middle bound (0.5)  | R$ 184,260.11 | 13.74% | 89.87% |
| Upper bound (0.75)  | R$ 198,208.98 | 17.96% | 91.33% |
## Data
Data was collected through web-scraping from one of the most representative marketplaces in Brazil the OLX Group.

Geographical features represent the minimum distance (in kilometers) from the nearest point (latitude and longitude) from a given vector or point and the point related to each property. The Haversine formula was used to calculate the distance between points.

Vectors of risky, green, inundation, and bus stop regions were extracted from [the city's Geographic Information Systems (GIS)](http://geoportal.pmf.sc.gov.br/downloads/camadas-em-sig-do-mapa). And the coordinates of health centers, markets, and schools were collected from [Google Maps](https://www.google.com.br/maps).



### Example of raw datapoint from listings
```
{
    "codigo": "1039411391",
    "descricao": "Apartamento para venda possui 93 metros quadrados com 3 quartos em Centro - Florianópolis ",
    "link": "https://sc.olx.com.br/florianopolis-e-regiao/imoveis/apartamento-para-venda-possui-93-metros-quadrados-com-3-quartos-em-centro-florianopolis-1039411391",
    "page": 1,
    "regiao": "centro",
    "Valor": "R$ 1.390.000",
    "Data": "Publicado em 28/09 às 16:54",
    "Categoria": "Apartamentos",
    "Tipo": "Venda - apartamento padrão",
    "Condomínio": "R$ 706",
    "IPTU": "R$ 1.373",
    "Área útil": "93m²",
    "Quartos": "3",
    "Banheiros": "3",
    "Vagas na garagem": "2",
    "Detalhes do imóvel": "Varanda, Mobiliado",
    "Detalhes do condominio": "Portaria, Salão de festas, Elevador",
    "CEP": "88015650",
    "Município": "Florianópolis",
    "Bairro": "Centro",
    "Logradouro": "Rua Henrique Bruggemann"
},
```
**New features:**

Extracted with CountVectorizer() from the `Detalhes do imóvel` and `Detalhes do condominio`:
- Binary: `piscina`, `academia`, `24h`
- Ordinal: `mobiliado`

Latitude and Longitude were created based on the address of each listing with the [geopy](https://github.com/geopy/geopy) package
- `lat`, `long`

Geographical features:
- From vectors: `risco`, `inundacao`, `vegetacao`, `onibus`
- From Points: `escolas`, `mercados`, `saude`
  
### Example of the datapoint from the final dataset
```
{'categoria': 'Apartamento',
  'condominio': 2600.0,
  'area': 520.0,
  'lat': -27.5890871,
  'long': -48.5586603,
  'quartos': 3.0,
  'banheiros': 5.0,
  'vagas_na_garagem': 3.0,
  'saude': 0.42,
  'mercados': 0.21,
  'escolas': 0.63,
  'onibus': 0.21,
  'vegetacao': 1.76,
  'risco': 1.84,
  'inundacao': 3.47,
  '24h': 0,
  'mobiliado': 2,
  'piscina': 1,
  'academia': 0,
  'valor': 2800000.0}
```
### Scraper

```
python src/scraper_script.py
```
## Setup

## Limitations and improvements

- Some addresses doesn't have the number of the street, therefore some locations are not exact
- Could do more text classification to create new features or improve others
- Improve scripts, readability, unit testing
- More extensive feature engineering
- The fitted model is too large, reduce it with feature selection
- Improve web-scraper to automate collection for each week ahead
- More hyperparameter tuning




## License
This repository is under an MIT License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ericyaang/instant-property-prediction/blob/main/LICENSE)