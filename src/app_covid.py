import streamlit as st
import pandas as pd
import numpy as np

from helpers import preprocessamento, download_url, load_dataframe, casos_por_obitos_do_estado, plot_timeseries_casos_por_obitos


def main():
    st.title('Análise de Registros de Caso e Óbitos de COVID-19 no Brasil')

    df = load_dataframe(output_path)
    df = preprocessamento(df)

    todos_os_estados   = sorted(df['estado'].dropna().unique().tolist())
    todos_os_estados.insert(0,'Brasil')
    estado_selecionado = st.sidebar.selectbox('Selecione o Estado', todos_os_estados, index=0,)

    df_estado = casos_por_obitos_do_estado(df, estado=estado_selecionado, casos_por_obitos=1)

    st.markdown('Abaixo um **dataframe**:')
    st.text_area('area51','aqui uma área de texto (sem formatação?)')
    # st.dataframe(df_estado.set_index('data').loc['2020-10':'2021-01'])
    
    figura = plot_timeseries_casos_por_obitos(df_estado)
    st.pyplot(fig=figura)

if __name__ == '__main__':
    url         = 'https://github.com/fontanads/bootcamp_dsa_2021/raw/main/data/HIST_PAINEL_COVIDBR_12jan2021.zip'    
    output_path = 'HIST_PAINEL_COVIDBR_12jan2021.zip'
    download_url(url, output_path) 
    main()
# "Comandos mágicos do streamlit"
# df_estado



