import haversine as hs
import joblib
from pathlib import Path
from config import config
import pandas as pd
import streamlit as st
import re
import geopandas as gp
import fiona
import numpy as np


def extract_string(str):
    return " ".join(re.findall("[a-zA-Z]+", str))


@st.cache(allow_output_mutation=True)
def processed_eda():
    import numpy as np

    df = pd.read_parquet(Path(config.DATA_PROCESSED_DIR, "data_cleaned.parquet"))
    df = df.drop_duplicates(subset=["codigo"])
    df = df.drop(columns=["codigo", "detalhes"], axis=1)
    df = df.dropna(subset=["valor", "area", "bairro", "risco", "quartos"])
    df["valor_m2"] = df["valor"] / df["area"]
    df = df[
        (df["valor_m2"] < df.valor_m2.quantile(0.999))
        & (df["valor_m2"] >= df.valor_m2.quantile(0.001))
    ]
    df = df[~(df["condominio"] > df["valor"] * 0.0025)]
    labels = ["0-50", "50-150", "150-250", "250-350", "350-450", "450-550", ">550"]
    intervals = [0, 50, 150, 250, 350, 450, 550, np.Inf]
    df["area_int"] = pd.cut(
        (df["area"]), bins=intervals, labels=labels, include_lowest=True
    )
    df.area_int = df.area_int.astype("category")
    df["area_int"] = df["area_int"].cat.reorder_categories(labels)
    df = df[~((df["area_int"] == "0-50") & (df["tipo"] == "apartamento cobertura"))]
    df.condominio = df.condominio.fillna(df.condominio.median())
    df.vagas_na_garagem = df.vagas_na_garagem.fillna(df.vagas_na_garagem.median())
    df.banheiros = df.banheiros.fillna(df.banheiros.median())
    return df


@st.cache(allow_output_mutation=True)
def load_data():
    return pd.read_parquet(Path(config.DATA_PROCESSED_DIR, "eda.parquet"))


@st.cache(allow_output_mutation=True)
def load_models():
    model_lower = joblib.load(Path(config.MODEL_REGISTRY, "Lower_final.bin"))
    model_median = joblib.load(Path(config.MODEL_REGISTRY, "Median_final.bin"))
    model_upper = joblib.load(Path(config.MODEL_REGISTRY, "Upper_final.bin"))
    return (model_lower, model_median, model_upper)


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

def get_geo_scores(df):
    from sklearn.preprocessing import MinMaxScaler

    mm_sc = MinMaxScaler()
    df_ = df[
        ["saude", "mercados", "escolas", "onibus", "vegetacao", "risco", "inundacao"]
    ].copy()
    df_["scores_scaled"] = df_.apply(set_scores, axis=1)
    scaled_score = df_["scores_scaled"].values.astype(float).reshape(-1, 1)
    scaled_array = mm_sc.fit_transform(scaled_score)
    prop_scaled = pd.DataFrame(
        scaled_array, columns=["scores_scaled"], index=(df.index)
    )
    return pd.concat([df, prop_scaled], axis=1)


def prepare_scores(df, bairro):
    df["bairro"] = bairro
    labels = ["0-50", "50-150", "150-250", "250-350", "350-450", "450-550", ">550"]
    intervals = [0, 50, 150, 250, 350, 450, 550, np.Inf]
    df["area_int"] = pd.cut(
        (df["area"]), bins=intervals, labels=labels, include_lowest=True
    )
    df.area_int = df.area_int.astype("category")
    df["area_int"] = df["area_int"].cat.reorder_categories(labels)
    return df


def set_score(row):
    score = (
        row["risco"] * -3
        + row["inundacao"] * -2
        + row["escolas"] * 1
        + row["mercados"] * 1
        + row["saude"] * 1.5
        + row["onibus"] * 1
        + row["vegetacao"] * 1
    )
    return score * -1



def set_local_score(row):
    score = (
        row["escolas"]
        + row["mercados"]
        + row["saude"]
        + row["onibus"]
        + row["vegetacao"]
    )
    return score * -1


def set_risco_score(row):
    score = row["risco"] * 3 + row["inundacao"] * 1.5
    return score


def add_geo_scores(df: pd.DataFrame, set_score, score_name: str, cols: list):
    from sklearn.preprocessing import MinMaxScaler

    mm_sc = MinMaxScaler()
    df_ = df[cols].copy()
    df_[score_name] = df_.apply(set_score, axis=1)
    scaled_score = df_[score_name].values.astype(float).reshape(-1, 1)
    scaled_array = mm_sc.fit_transform(scaled_score)
    prop_scaled = pd.DataFrame(scaled_array, columns=[score_name], index=(df_.index))
    return pd.concat([df, prop_scaled], axis=1)


def get_bairro_layer():
    fiona.supported_drivers["KML"] = "rw"
    saude_layer = gp.read_file("data\\geo\\saude-layers.kml", driver="KML")
    saude_layer["Name"] = saude_layer["Name"].str.replace("CS", "").str.strip()
    saude_layer["Name"] = saude_layer["Name"].str.replace("JURERE", "JURERÊ")
    saude_layer["Name"] = saude_layer["Name"].str.replace("MONTE SERRAT", "CENTRO")
    saude_layer["Name"] = saude_layer["Name"].str.replace(
        "SANTO ANTONIO DE LISBOA", "SANTO ANTÔNIO DE LISBOA"
    )
    saude_layer = saude_layer.iloc[:49, :]
    saude_layer.iloc[(43, 2)] = saude_layer.query(
        'Name == "SAPÉ"|Name =="JARDIM ATLÂNTICO"'
    ).unary_union
    saude_layer = saude_layer[~(saude_layer["Name"] == "SAPÉ")]
    layer_dict = {
        "ALTO RIBEIRÃO": "RIBEIRÃO DA ILHA",
        "ARMAÇÃO": "ARMAÇÃO DO PÂNTANO DO SUL",
        "CAIEIRA DA BARRA DO SUL": "RIBEIRÃO DA ILHA",
        "CANTO DA LAGOA": "LAGOA DA CONCEIÇÃO",
        "COSTA DA LAGOA": "LAGOA DA CONCEIÇÃO",
        "FAZENDA DO RIO TAVARES": "RIO TAVARES",
        "INGLESES": "INGLESES DO RIO VERMELHO",
        "NOVO CONTINENTE": "ESTREITO",
        "PRAINHA": "JOSÉ MENDES",
        "RIO VERMELHO": "SÃO JOÃO DO RIO VERMELHO",
        "SANTINHO": "INGLESES DO RIO VERMELHO",
        "TAPERA": "TAPERA DA BASE",
        "VILA APARECIDA": "COQUEIROS",
    }
    saude_layer["Name"] = saude_layer["Name"].replace(layer_dict)
    saude_layer = saude_layer.rename(columns={"Name": "bairro", "index": "count"})
    saude_layer = saude_layer.dissolve(by="bairro", aggfunc="sum")
    saude_layer = saude_layer.reset_index()
    return saude_layer


def get_bairro_stats(df):
    df_bairros = pd.DataFrame(df.groupby("bairro")["risco_score"].mean())
    df_bairros["local_score"] = df.groupby("bairro")["local_score"].mean()
    df_bairros["all_score"] = df.groupby("bairro")["all_score"].mean()
    df_bairros["valor_m2"] = df.groupby("bairro")["valor_m2"].median()
    df_bairros["valor"] = df.groupby("bairro")["valor"].median()
    return df_bairros.reset_index()
