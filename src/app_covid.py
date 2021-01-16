import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from zipfile import ZipFile
from StringIO import StringIO
import urllib2

# from helpers import preprocessamento, download_url, load_dataframe, casos_por_obitos_do_estado, plot_timeseries_casos_por_obitos

# @st.cache
# def carrega_dataframe(output_path):
#     df = load_dataframe(output_path)
#     df = preprocessamento(df)
#     todos_os_estados   = sorted(df['estado'].dropna().unique().tolist())
#     todos_os_estados.insert(0,'Brasil')
#     return df, todos_os_estados

# def carrega_dados_e_plota(df, estado_selecionado, casos_por_obitos='Casos por óbito'):
#     df_estado = casos_por_obitos_do_estado(df, estado=estado_selecionado, casos_por_obitos=casos_por_obitos)
#     figura = plot_timeseries_casos_por_obitos(df_estado, casos_por_obitos=casos_por_obitos)
#     return figura


def download_url(url, save_path):
    urllib.request.urlretrieve(url, save_path)


def main():
    df = pd.read_csv(csv_file, sep=';')
    # df, todos_os_estados = carrega_dataframe(output_path)

    st.title('Análise de Registros de Caso e Óbitos de COVID-19 no Brasil')
    # st.markdown('Abaixo um **dataframe**:')
    # st.text_area('area51','aqui uma área de texto (sem formatação?)')
    # st.dataframe(df_estado.set_index('data').loc['2020-10':'2021-01'])

    # casos_por_obitos =   st.sidebar.selectbox('Selecione o Estado', ['Casos por óbito','Óbitos por caso'], index=0)
    # estado_selecionado = st.sidebar.selectbox('Selecione o Estado', todos_os_estados, index=0)

    # figura = carrega_dados_e_plota(df, estado_selecionado, casos_por_obitos=casos_por_obitos)
    # st.pyplot(fig=figura)

if __name__ == '__main__':
    url         = 'https://github.com/fontanads/bootcamp_dsa_2021/raw/main/data/HIST_PAINEL_COVIDBR_12jan2021.zip'    
    filename    = 'HIST_PAINEL_COVIDBR_12jan2021.csv'
    r = urllib2.urlopen(url).read()
    zip_file = ZipFile(StringIO(r))
    csv_file = zip_file.open(filename)
    main()