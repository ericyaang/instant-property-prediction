import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import streamlit.components.v1 as components
from src import app_utils

with open('src\style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

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

    lower_value_m2 = int(lower_value / df_new['area'])
    median_value_m2 = int(median_value / df_new['area'])
    upper_value_m2 = int(upper_value / df_new['area'])
    
    # ******* Create new dataset with df_new to concatenate other features *******
    
    df_new['bairro'] = bairro
    df_new = app_utils.prepare_scores(df_new, bairro)
    df_new['valor'] = median_value
    df_new['valor_m2'] = median_value / df['area']
    
    df_all = pd.concat([df_new, df], join='inner')

    ## add scores
    df_all = app_utils.add_geo_scores(df_all, set_score=app_utils.set_score, score_name='global_score', cols=['saude', 'mercados', 'escolas', 'onibus', 'vegetacao', 'risco',
       'inundacao'])

    df_all = app_utils.add_geo_scores(df_all, set_score=app_utils.set_local_score, score_name='local_score', cols=['saude', 'mercados', 'escolas', 'onibus', 'vegetacao']) 

    df_all = app_utils.add_geo_scores(df_all, set_score=app_utils.set_risco_score, score_name='risco_score', cols=['risco', 'inundacao'])

    # columns for price
    col1, col2, col3 = st.columns(3)
    
    # settings
    # hide metric arrow
    st.write(
        """
        <style>
        [data-testid="stMetricDelta"] svg {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # remove 'Made with Streamlit' footer MainMenu {visibility: hidden;}
    hide_streamlit_style = """
                <style>
                footer {visibility: hidden;}
                </style>
                """

    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    with col1:
        st.header("")
        st.metric('Mínimo', (f"R$ {lower_value:,.0f}").replace(',','.'), (f"R$ {lower_value_m2:,.0f} por m²").replace(',','.'), delta_color='off')
        st.caption('Média do Bairro: ' + str(int(df_all.query(f'bairro == "{bairro}"').groupby('bairro')['valor_m2'].mean()[0]*100)))

    with col2:
        st.header("")
        st.metric('Médio', (f"R$ {median_value:,.0f}").replace(',','.'), (f"R$ {median_value_m2:,.0f} por m²").replace(',','.'), delta_color='off')
        #st.caption('Valor médio no bairro: ' + str(int(df_all.query(f"bairro == '{df_all['bairro'].iloc[0,]}' & area_int == '{df_all['area_int'].iloc[0,]}'").groupby('bairro')['valor_m2'].mean()[0])))
    with col3:
        st.header("")
        st.metric('Máximo', (f"R$ {upper_value:,.0f}").replace(',','.'), (f"R$ {upper_value_m2:,.0f} por m²").replace(',','.'), delta_color='off')
    
    import numpy as np
    with st.container():     
        st.header("Ranking de proximidade")
        st.write('Ranking de 0 a 100.')

        col1, col2, col3 = st.columns(3)        
        with col1:
            st.metric('Global Score', int(df_all['global_score'].iloc[0,] * 100), delta_color='normal')
            st.caption('Média do Bairro: ' + str(int(df_all.query(f'bairro == "{bairro}"').groupby('bairro')['risco_score'].mean()[0]*100)))

        with col2:
            st.metric('Local Score', int(df_all['local_score'].iloc[0,] * 100), delta_color='normal')
            st.caption('Média do Bairro: ' + str(int(df_all.query(f'bairro == "{bairro}"').groupby('bairro')['local_score'].mean()[0]*100)))
        with col3:
            st.metric('Risco Score', int(df_all['risco_score'].iloc[0,]* 100), delta_color='normal')
            st.caption('Média do Bairro: ' + str(int(df_all.query(f'bairro == "{bairro}"').groupby('bairro')['risco_score'].mean()[0]*100)))
        # showing the maps
        import pydeck as pdk        
        sf_initial_view = pdk.ViewState(
            latitude=float(lat),
            longitude=float(long),
            zoom=11,
            pitch=30
        )
        polygon_layer = pdk.Layer(
        "PolygonLayer",
        df_all,
        id="geojson",
        opacity=0.5,
        stroked=False,
        get_polygon="coordinates",
        filled=True,
        extruded=True,
        wireframe=True,
        get_elevation="floorsize_m2",
        get_fill_color="fill_color",
        get_line_color=[255, 255, 255],
        auto_highlight=True,
        pickable=True,
        )   

        COLOR_BREWER_RED_SCALE = [
        [233, 255, 214],
        [209, 254, 175],
        [196, 254, 154],
        [177, 251, 136],]

        hx_layer = pdk.Layer(
            'HexagonLayer',
            data = df_all[['long', 'lat','valor_m2']],
            get_position = ['long', 'lat'],
            get_elevation='valor_m2 / 20',
            elevation_range=[0, 2000],
            color_range=[
                [197, 249, 215],
                [247, 212, 134],
                [242, 122, 125]
            ],
            elevation_scale=2,
            #pickable=True, # enable text labels
            radius=50,
            extruded=True)


        tooltip={"text": "Count: {valor_m2}"}

        map = pdk.Deck(
        map_style='light',
        initial_view_state=sf_initial_view,
        layers = [hx_layer],
        tooltip=tooltip
        )
        components.html(map.to_html(as_string=True), height=400)        

else:
    st.write("Para começar a avaliação preencha o questionário ao lado.")
    st.stop()
