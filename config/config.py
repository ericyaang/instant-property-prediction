from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.absolute()

CONFIG_DIR = Path(BASE_DIR, "config")
DATA_RAW_DIR = Path(BASE_DIR, "data/raw")
DATA_GEO_DIR = Path(BASE_DIR, "data/geo")
DATA_PROCESSED_DIR = Path(BASE_DIR, "data/processed")
STORES_DIR = Path(BASE_DIR, "stores")
MODEL_REGISTRY = Path(STORES_DIR, "model")
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)

# Local stores
MODEL_REGISTRY = Path(STORES_DIR, "model")
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)

selected_cols = [
    'categoria',
    'condominio', 'area',
    'lat', 'long',
    'quartos', 'banheiros', 'vagas_na_garagem',
    'saude', 'mercados', 'escolas', 'onibus', 'vegetacao', 'risco', 'inundacao',
    '24h', 'mobiliado', 'piscina', 'academia'
]

int_cols = ['quartos', 'banheiros', 'vagas_na_garagem']
cont_cols = ['condominio', 'area', 'lat', 'long', 'saude', 'mercados', 'escolas', 'onibus', 'vegetacao', 'risco', 'inundacao']
cat_cols = ['categoria']
ord_cols = ['mobiliado']
bi_cols = ['24h', 'piscina', 'academia']

params_search={
    'max_depth': [-1, 5, 10, 25],
    'learning_rate': [0.01, 0.05, 0.1, 0.25],
    'num_leaves': [10, 31, 50],
    'n_estimators': [10, 100, 250],
    'boosting_type': ["gbdt"],
    'reg_alpha': [0, 0.01, 0.05, 0.1],
    'reg_lambda': [0, 0.01, 0.05, 0.1]}