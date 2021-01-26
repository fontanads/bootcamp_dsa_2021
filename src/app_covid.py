# Primeira versão com estatísticas descritivas de um dataset fixo

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

import plotly.offline as pyo
import plotly.graph_objs as go

from zipfile import ZipFile
from io import BytesIO
import urllib3

from helpers import preprocessamento, casos_por_obitos_do_estado, plot_timeseries_casos_por_obitos

def carrega_dataframe(csv_file):
    df = pd.read_csv(csv_file, sep=';')
    df = preprocessamento(df)
    todos_os_estados   = sorted(df['estado'].dropna().unique().tolist())
    todos_os_estados.insert(0,'Brasil')
    return df, todos_os_estados

def carrega_dados_estado(df, estado_selecionado, casos_por_obitos='Casos por óbito'):
    df_estado = casos_por_obitos_do_estado(df, estado=estado_selecionado, casos_por_obitos=casos_por_obitos)
    return df_estado

def download_url(url, filename):
    http = urllib3.PoolManager() 
    r = http.request('GET', url)
    zip_file = ZipFile(BytesIO(r.data))
    csv_file = zip_file.open(filename)
    return csv_file

def selecoes_usuario_var_e_estado():
    var    = st.sidebar.selectbox('Selecione o Estado', ['Casos por óbito','Óbitos por caso'], index=0)
    estado = st.sidebar.selectbox('Selecione o Estado', todos_os_estados, index=0)
    return (var, estado)

def main():

    st.title('Análise de Registros de Caso e Óbitos de COVID-19 no Brasil')
    # st.markdown('Abaixo um **dataframe**:')
    # st.text_area('area51','aqui uma área de texto (sem formatação?)')
    # st.dataframe(df_estado.set_index('data').loc['2020-10':'2021-01'])

    casos_por_obitos, estado_selecionado = selecoes_usuario_var_e_estado()

    df_estado = carrega_dados_estado(df, estado_selecionado, casos_por_obitos=casos_por_obitos)
    figura = plot_timeseries_casos_por_obitos(df_estado, casos_por_obitos=casos_por_obitos)

    st.plotly_chart(figura)

if __name__ == '__main__':
    url         = 'https://github.com/fontanads/bootcamp_dsa_2021/raw/main/data/HIST_PAINEL_COVIDBR_12jan2021.zip'    
    filename    = 'HIST_PAINEL_COVIDBR_12jan2021.csv'
    csv_file = download_url(url, filename)    
    df, todos_os_estados = carrega_dataframe(csv_file)

    main()