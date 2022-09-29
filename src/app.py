import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import streamlit.components.v1 as components
from src import app_utils

df = app_utils.load_data()
model_lower, model_median, model_upper = app_utils.load_models()

with st.sidebar:
    st.sidebar.header("Complete o questionário")
    st.sidebar.warning("Válido apenas para a cidade de Florianópolis.")
    endereco = st.sidebar.selectbox(
        "Rua",
        (df.logradouro.str.replace("\\d+|/| a ", "", regex=True).unique()),
        index=1064,
    )
    numero = st.sidebar.number_input(
        "Número", min_value=1, max_value=100000, value=142, step=1
    )
    bairro = st.sidebar.selectbox("Bairro", (df.bairro.unique()), index=1)
    address = endereco + ", " + str(numero) + ", " + bairro
    geolocator = Nominatim(user_agent="GTA Lookup")
    geocode = RateLimiter((geolocator.geocode), min_delay_seconds=1)
    location = geolocator.geocode(address + ", Florianópolis, Santa Catarina, Brasil")
    if location is not None:
        lat = str(location.latitude)
        long = str(location.longitude)
    else:
        st.error("Oops! Endereço incorreto. Por favor, corrija e tente novamente. ")
        st.stop()
    categoria = st.sidebar.selectbox("Categoria", ["Apartamento", "Casa"])
    area = st.sidebar.number_input(
        "Área (m²)", 30, 5000, help="Área privativa ou construída", value=50
    )
    condominio = st.sidebar.number_input("Condomínio", 0, 1000000, value=0)
    quartos = st.sidebar.select_slider("Quartos", ["1", "2", "3", "4", "5"])
    banheiros = st.sidebar.select_slider("Banheiros", ["1", "2", "3", "4", "5"])
    garagem = st.sidebar.select_slider("Garagem", ["0", "1", "2", "3", "4", "5"])
    mobiliado = st.sidebar.select_slider("Mobiliado?", ["não", "parcialmente", "sim"])
    if mobiliado == "não":
        mobiliado = 0
    else:
        if mobiliado == "parcialmente":
            mobiliado = 1
        else:
            if mobiliado == "sim":
                mobiliado = 2

    seguranca_24 = st.sidebar.checkbox("Segurança 24 horas")
    if seguranca_24 is True:
        seguranca_24 = 1
    else:
        if seguranca_24 is False:
            seguranca_24 = 0
    piscina = st.sidebar.checkbox("Piscina")
    if piscina is True:
        piscina = 1
    else:
        if piscina is False:
            piscina = 0
    academia = st.sidebar.checkbox("Academia")
    if academia is True:
        academia = 1
    else:
        if academia is False:
            academia = 0
    avaliar = st.sidebar.button("Avaliar")
if avaliar:
    st.title("Quanto vale o meu imóvel?")
    # criar dataframe
    input_df = pd.DataFrame(
        [
            {
                "mobiliado": mobiliado,
                "quartos": quartos,
                "banheiros": banheiros,
                "vagas_na_garagem": garagem,
                "condominio": condominio,
                "area": area,
                "lat": lat,
                "long": long,
                "saude": None,
                "mercados": None,
                "escolas": None,
                "onibus": None,
                "vegetacao": None,
                "risco": None,
                "inundacao": None,
                "categoria": categoria,
                "24h": seguranca_24,
                "piscina": piscina,
                "academia": academia,
            }
        ]
    )

    # add geo features
    df_new = app_utils.get_geo(input_df)

    # load models
    lower_value = int(model_lower.predict(df_new))
    median_value = int(model_median.predict(df_new))
    upper_value = int(model_upper.predict(df_new))

    # kpi card html
    kpi_html = f"""
        <script>
            document.head.innerHTML +=
            '<link href="https://fonts.googleapis.com/css2?family=Open+Sans&display=swap" rel="stylesheet">'
        </script>
            <div style="font-family: 'Open Sans';
            display: flex;
            flex-direction: row;
            color: #FFFFFF;
            letter-spacing: 0.2em;
            margin-bottom: auto;">            
        <div style="margin: auto;">
            <div style="color: #3A3535;">
            Mínimo
            </div>
            <div style="color: #232020; font-size: 1.2em;">
            {(f"R$ {lower_value:,.0f}").replace(',','.')}
            </div>
        </div>
        <div style="margin: auto;">
            <div style="color: #3A3535;">
            Valor Aimob
            </div>
            <div style="color: #232020; font-size: 2rem;">
            {(f"R$ {median_value:,.0f}").replace(',','.')}
            </div>
        </div>      
            <div style="margin: auto;">
                <div style="color: #3A3535;">
                Máximo
                </div>
                <div style="color: #232020; font-size: 1.2em;">
                {(f"R$ {upper_value:,.0f}").replace(',','.')}
                </div>          
        </div>
        </div>
    """
    components.html(kpi_html, height=100)
else:
    st.write("Para começar a avaliação preencha o questionário ao lado.")
    st.stop()
