import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np

from zipfile import ZipFile
import requests 

def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def load_dataframe(caminho, verbose=True):
    arquivo = ZipFile(caminho)
    if verbose:
        for i,f in enumerate(arquivo.namelist()):
            print(f' ({i}) {f}')
    file_to_load_df = arquivo.namelist()[0]

    df = pd.read_csv(arquivo.open(file_to_load_df), sep=';')
    return df

def preprocessamento(df):
    df['data'] = pd.to_datetime(df['data'])
    df['obitos_por_casos_novos']     = np.log10(df['obitosNovos']/df['casosNovos'])
    df['obitos_por_casos_acumulado'] = np.log10(df['obitosAcumulado']/df['casosAcumulado'])
    df['casos_por_obitos_novos']     = np.log10(df['casosNovos']/df['obitosNovos'])
    df['casos_por_obitos_acumulado'] = np.log10(df['casosAcumulado']/df['obitosAcumulado'])
    return df

def casos_por_obitos_do_estado(df, estado='RS', casos_por_obitos = 1):
    if estado == 'Brasil':
        df_estado = df[df['regiao']==estado]  
    else:
        df_estado = df[df['estado']==estado] 
        
    df_estado = df_estado[df_estado['codmun'].isna()]
    
    if casos_por_obitos==1:
        value_vars = ['casos_por_obitos_novos', 'casos_por_obitos_acumulado']
    else:
        value_vars = ['obitos_por_casos_novos', 'obitos_por_casos_acumulado']
    df_estado = df_estado.melt(id_vars='data', value_vars=value_vars).rename_axis(estado, axis='columns') # 'casosAcumulado' 'obitosAcumulado'
    return df_estado

def plot_timeseries_casos_por_obitos(df_estado):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11,7))
    sns_fig = sns.lineplot(data=df_estado, x='data', y='value', hue='variable', ax=ax)

    ax.set_ylabel('Casos por Óbito')
    ax.legend(['Novos','Acumulados'])
    ax.set_title(f'Série para o {df_estado.axes[1].name}')
    ax.set_yticks(ticks=range(5))
    ax.set_yticklabels(labels=10**np.array(range(5)))

    return fig

    # 
    # 

    # 
    # plt.show()

    return sns_fig