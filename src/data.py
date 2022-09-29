from pathlib import Path
from config import config
from sklearn.compose import make_column_transformer
from src import utils
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer
import re
import feature_engine.outliers as feo
import haversine as hs, unicodedata
from datetime import datetime as dt
import dask.dataframe as dd
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_data():
    return (
        dd.read_csv(
            (Path(config.DATA_RAW_DIR, "raw_*_*-*.csv")),
            dtype={
                "Banheiros": "object",
                "Quartos": "object",
                "Vagas na garagem": "object",
                "Área construída": "object",
            },
        )
        .compute()
        .drop_duplicates("codigo", keep="last")
    )


def process_pandas(df: pd.DataFrame):
    def start_pipeline(df_: pd.DataFrame):
        return df_.copy()

    def normalize(string: str):
        normalized = unicodedata.normalize("NFD", string)
        return re.sub("[\\u0300-\\u036f]", "", normalized).casefold()

    def clean_names(df_):
        return (
            df_.rename(columns=(lambda c: c.replace(" ", "_")))
            .rename(columns=(lambda c: normalize(c)))
            .assign(
                bairro=(lambda df_: df_.bairro.astype("category")),
                regiao=(lambda df_: df_.regiao.astype("category")),
                logradouro=(
                    lambda df_: df_.logradouro.str.strip()
                    .str.replace(
                        "ao fim|- lado ímpar|- lado par|- de|- até", "", regex=True
                    )
                    .astype("category")
                ),
                tipo=(
                    lambda df_: df_.tipo.str.strip()
                    .str.replace("Venda - ", "")
                    .astype("category")
                ),
                categoria=(
                    lambda df_: df_.categoria.str.strip()
                    .str.replace("Casas", "Casa")
                    .str.replace("Apartamentos", "Apartamento")
                    .astype("category")
                ),
            )
            .drop(columns=["link", "page", "municipio", "iptu"])
        )

    def extract_time(df_: pd.DataFrame):
        def converter(string: str):
            match = re.search("\\d{2}/\\d{2}", string)
            return dt.strptime(match.group(), "%d/%m").replace(year=(dt.today().year))

        df_["data"] = df_["data"].apply(lambda row: converter(row))
        return df_

    def extract_numeric(df_: pd.DataFrame):
        num_cols = [
            "valor",
            "condominio",
            "area_util",
            "quartos",
            "banheiros",
            "vagas_na_garagem",
            "area_construida",
        ]
        df_[num_cols] = df_[num_cols].apply(
            lambda row: row.str.replace("[^0-9]", "", regex=True)
        )
        df_[num_cols] = df_[num_cols].apply((pd.to_numeric), errors="coerce")
        return df_

    def new_columns(df_: pd.DataFrame):
        def uniquify(string: str):
            output = []
            seen = set()
            for word in string.split(", "):
                if word not in seen:
                    output.append(word)
                    seen.add(word)
            else:
                return ", ".join(output)

        var_area = ["area_construida", "area_util"]
        var_detalhes = ["detalhes_do_condominio", "detalhes_do_imovel"]
        df_["area"] = df_.area_util.fillna(df_.area_construida)
        df_[var_detalhes] = df_[var_detalhes].fillna("None")
        df_["detalhes"] = (
            df_["detalhes_do_condominio"] + ", " + df_["detalhes_do_imovel"]
        )
        df_["detalhes"] = df_["detalhes"].apply(lambda row: uniquify(row))
        df_ = df_.drop(columns=(var_detalhes + var_area), axis=1)
        return df_

    def filters(df_):
        df_.descricao = df_.descricao.apply(lambda x: normalize(x))
        df_ = df_[
            ~df_.descricao.str.contains(
                "aluga|aluguel|comercial|construçao|terreno|loteamento|planta"
            )
        ]
        df_ = df_[~df_.bairro.str.contains("Área Rural de Florianópolis")]
        df_ = df_[~df_.logradouro.isna()]
        df_ = df_[~df_.logradouro.str.contains("Rodovia BR-282")]
        df_ = df_.drop("descricao", axis=1)
        return df_

    def add_coordinates(df_: pd.DataFrame):
        coord = pd.read_parquet(Path(config.DATA_GEO_DIR, "coordenadas.parquet"))
        lat_dict = dict(zip(coord.logradouro, coord.lat))
        long_dict = dict(zip(coord.logradouro, coord.long))
        df_["lat"] = df_["logradouro"].map(lat_dict).astype("float")
        df_["long"] = df_["logradouro"].map(long_dict).astype("float")
        df_ = df_.dropna(subset=["lat", "long"])
        return df_

    def add_geo_features(df_: pd.DataFrame):
        spatial_features = pd.read_parquet(
            Path(config.DATA_GEO_DIR, "geo-features.parquet")
        )
        geo_features = [
            "saude",
            "mercados",
            "escolas",
            "onibus",
            "vegetacao",
            "risco",
            "inundacao",
        ]
        for feature in geo_features:
            new_dict = dict(zip(spatial_features.logradouro, spatial_features[feature]))
            df_[feature] = df_["logradouro"].map(new_dict)
            df_[feature] = df_[feature].astype("float")
        else:
            df_ = df_.dropna(subset=[feature])
            return df_

    def replace_null(df_: pd.DataFrame):
        params_fp = Path(config.CONFIG_DIR, "params.json")
        params = utils.load_dict(filepath=params_fp)
        df_.loc[(df_["area"] >= params["max_area"], ["area"])] = np.nan
        df_.loc[(df_["area"] < params["min_area"], ["area"])] = np.nan
        df_ = df_[~df_["area"].isnull()]
        df_ = df_[~(df_["valor"] < params["min_area"] * params["valor_m2_min"])]
        df_ = df_[~(df_["valor"] > params["max_area"] * params["valor_m2_max"])]
        df_ = df_[~df_["valor"].isna()]
        for col in ("valor", "quartos", "banheiros", "area"):
            df_[col] = df_[col].apply(lambda x: np.nan if x == 0 else x)
        else:
            limit_low = (
                df_["area"] * params["valor_m2_min"] * params["max_tax_cond"] / 2
            )
            limit_high = df_["area"] * params["valor_m2_max"] * params["max_tax_cond"]
            df_["condominio"] = np.where(
                df_["condominio"] >= limit_high, np.nan, df_["condominio"]
            )
            df_.loc[
                (
                    (df_["categoria"] == "Apartamento")
                    & (df_["condominio"] < limit_low),
                    ["condominio"],
                )
            ] = np.nan
            df_.loc[
                (
                    (df_["categoria"] == "Casa")
                    & (df_["condominio"] != 0)
                    & (df_["condominio"] < limit_low),
                    ["condominio"],
                )
            ] = np.nan
            df_ = df_[~(df_["valor"] < params["min_area"] * params["valor_m2_min"])]
            df_ = df_[~(df_["valor"] > params["max_area"] * params["valor_m2_max"])]
            df_ = df_[~df_["valor"].isna()]
            return df_

    return (
        df.pipe(start_pipeline)
        .pipe(clean_names)
        .pipe(extract_time)
        .pipe(extract_numeric)
        .pipe(new_columns)
        .pipe(filters)
        .pipe(add_geo_features)
        .pipe(add_coordinates)
        .pipe(replace_null)
        .set_index("codigo")
    )


def get_data_splits(df: pd.DataFrame):
    params_fp = Path(config.CONFIG_DIR, "params.json")
    params = utils.load_dict(filepath=params_fp)
    train, test = train_test_split(
        df, train_size=(params["train_size"]), random_state=(params["seed"])
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    return (train, test)


def prepare_cv(X_train: pd.DataFrame, X_test: pd.DataFrame):
    params_fp = Path(config.CONFIG_DIR, "params.json")
    params = utils.load_dict(filepath=params_fp)
    cv = CountVectorizer(
        lowercase=True,
        max_features=20,
        binary=True,
        stop_words=(params["step_words"]),
        analyzer="word",
        ngram_range=(1, 1),
    )
    X_train_cv = pd.DataFrame(
        (cv.fit_transform(X_train["detalhes"]).toarray()),
        columns=(cv.get_feature_names_out()),
        index=(X_train.index),
    )
    X_test_cv = pd.DataFrame(
        (cv.transform(X_test["detalhes"]).toarray()),
        columns=(cv.get_feature_names_out()),
        index=(X_test.index),
    )
    for X in (X_train_cv, X_test_cv):
        X["mobiliado"] = np.where(
            X["mobiliado"] == 1, 2, np.where(X["armários"] > 0, 1, 0)
        )
    else:
        selected_cols = ["24h", "mobiliado", "piscina", "academia"]
        X_train_trans = pd.concat([X_train, X_train_cv[selected_cols]], axis=1)
        X_test_trans = pd.concat([X_test, X_test_cv[selected_cols]], axis=1)
        X_train_trans = X_train_trans.drop(columns=["detalhes"], axis=1)
        X_test_trans = X_test_trans.drop(columns=["detalhes"], axis=1)
        return (X_train_trans, X_test_trans)


def impute_data(X_train: pd.DataFrame, X_test: pd.DataFrame):
    params_fp = Path(config.CONFIG_DIR, "params.json")
    params = utils.load_dict(filepath=params_fp)
    imputer = make_column_transformer(
        (SimpleImputer(strategy="constant", fill_value=0), params["ord_cols"]),
        (
            SimpleImputer(strategy="most_frequent"),
            params["int_cols"] + params["bi_cols"],
        ),
        (SimpleImputer(strategy="median"), params["cont_cols"]),
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    X_train_imputed = pd.DataFrame(
        (imputer.fit_transform(X_train)),
        columns=(imputer.get_feature_names_out()),
        index=(X_train.index),
    ).astype(X_train.dtypes)
    X_test_imputed = pd.DataFrame(
        (imputer.transform(X_test)),
        columns=(imputer.get_feature_names_out()),
        index=(X_test.index),
    ).astype(X_train.dtypes)
    return (X_train_imputed, X_test_imputed)


def split_data(df: pd.DataFrame):
    params_fp = Path(config.CONFIG_DIR, "params.json")
    params = utils.load_dict(filepath=params_fp)
    train, df_ = train_test_split(
        df, train_size=(params["train_size"]), random_state=(params["seed"])
    )
    valid, test = train_test_split(
        df_, train_size=0.5, random_state=(params["seed"] + 1)
    )
    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    test = test.reset_index(drop=True)
    return (train, valid, test)


def winsorizer(df):
    df["valor_m2"] = df["valor"] / df["area"]
    trimmer = feo.Winsorizer(
        capping_method="quantiles",
        tail="both",
        fold=0.001,
        variables=["valor_m2"],
        add_indicators=True,
    )
    df_t = trimmer.fit_transform(df)
    df_t = df_t[~(df_t.valor_m2_right == 1)]
    df_t = df_t[~(df_t.valor_m2_left == 1)]
    df_t = df_t.drop(columns=["valor_m2"], axis=1)
    return df_t


def distance_from(loc1: str, loc2: str):
    dist = hs.haversine(loc1, loc2)
    return round(dist, 2)


def get_geo(df: pd.DataFrame):
    saude_df = pd.read_parquet(
        (Path(config.DATA_GEO_DIR, "saude_df.parquet")), engine="pyarrow"
    )
    mercado_df = pd.read_parquet(
        (Path(config.DATA_GEO_DIR, "mercado_df.parquet")), engine="pyarrow"
    )
    escola_df = pd.read_parquet(
        (Path(config.DATA_GEO_DIR, "escola_df.parquet")), engine="pyarrow"
    )
    onibus_df = pd.read_parquet(
        (Path(config.DATA_GEO_DIR, "onibus_df.parquet")), engine="pyarrow"
    )
    risco_df = pd.read_parquet(
        (Path(config.DATA_GEO_DIR, "risco_df.parquet")), engine="pyarrow"
    )
    inundacao_df = pd.read_parquet(
        (Path(config.DATA_GEO_DIR, "inundacao_df.parquet")), engine="pyarrow"
    )
    vegetacao_df = pd.read_parquet(
        (Path(config.DATA_GEO_DIR, "vegetacao_df.parquet")), engine="pyarrow"
    )
    df["lat"] = df["lat"].astype("float")
    df["long"] = df["long"].astype("float")
    df["coor"] = list(zip(df.lat, df.long))
    df["saude"] = df["coor"].apply(
        lambda x: [distance_from(i, x) for i in saude_df["coor"]]
    )
    df["saude"] = df["saude"].apply(lambda x: min(x))
    df["mercados"] = df["coor"].apply(
        lambda x: [distance_from(i, x) for i in mercado_df["coor"]]
    )
    df["mercados"] = df["mercados"].apply(lambda x: min(x))
    df["escolas"] = df["coor"].apply(
        lambda x: [distance_from(i, x) for i in escola_df["coor"]]
    )
    df["escolas"] = df["escolas"].apply(lambda x: min(x))
    df["onibus"] = df["coor"].apply(
        lambda x: [distance_from(i, x) for i in onibus_df["coor"]]
    )
    df["onibus"] = df["onibus"].apply(lambda x: min(x))
    df["vegetacao"] = df["coor"].apply(
        lambda x: [distance_from(i, x) for i in vegetacao_df["coor"]]
    )
    df["vegetacao"] = df["vegetacao"].apply(lambda x: min(x))
    df["risco"] = df["coor"].apply(
        lambda x: [distance_from(i, x) for i in risco_df["coor"]]
    )
    df["risco"] = df["risco"].apply(lambda x: min(x))
    df["inundacao"] = df["coor"].apply(
        lambda x: [distance_from(i, x) for i in inundacao_df["coor"]]
    )
    df["inundacao"] = df["inundacao"].apply(lambda x: min(x))
    df = df.drop("coor", axis=1)
    return df


def set_scores(row):
    score = (
        row["risco"] * -2
        + row["inundacao"] * -1.5
        + row["escolas"] * 1
        + row["mercados"] * 1
        + row["saude"] * 1.5
        + row["onibus"] * 1
        + row["vegetacao"] * 1
    )
    return score * -1
