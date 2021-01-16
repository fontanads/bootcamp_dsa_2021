import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np

import plotly.offline as pyo
import plotly.graph_objs as go

def preprocessamento(df):
    df['data'] = pd.to_datetime(df['data'])
    df['obitos_por_casos_novos']     = 100*(df['obitosNovos']/df['casosNovos'])
    df['obitos_por_casos_acumulado'] = 100*(df['obitosAcumulado']/df['casosAcumulado'])
    df['casos_por_obitos_novos']     = (df['casosNovos']/df['obitosNovos'])
    df['casos_por_obitos_acumulado'] = (df['casosAcumulado']/df['obitosAcumulado'])
    return df

def casos_por_obitos_do_estado(df, estado='RS', casos_por_obitos = 1):
    if estado == 'Brasil':
        df_estado = df[df['regiao']==estado]  
    else:
        df_estado = df[df['estado']==estado] 
    df_estado = df_estado[df_estado['codmun'].isna()]
    if casos_por_obitos=='Casos por óbito':
        value_vars = {'casos_por_obitos_novos':'novos', 'casos_por_obitos_acumulado':'acumulado'}
    else:
        value_vars = {'obitos_por_casos_novos':'novos', 'obitos_por_casos_acumulado':'acumulado'}
    df_estado = df_estado.melt(id_vars='data', value_vars=list(value_vars.keys())).rename_axis(estado, axis='columns').replace({'variable':value_vars}) 
    return df_estado

def plot_timeseries_casos_por_obitos(df_estado, casos_por_obitos=1):
    if casos_por_obitos=='Casos por óbito':
        ylabel = 'Casos por Óbito'
        y_axis_type="log"
    else:
        ylabel = 'Óbitos a cada 100 casos'
        y_axis_type="linear"
    title = f'Série para o {df_estado.axes[1].name}'
    xlabel = 'Data'

    marker_dict_0 = dict(
        size=5,
        color ='rgb(54,28,100)',
        symbol='hexagon2',
        line={'width':1}
        )
    
    marker_dict_1 = dict(
        size=5,
        color ='rgb(150,28,64)',
        symbol='hexagon2',
        line={'width':1}
        )

    # trace 0
    x_values = df_estado['data'] 
    y_values = df_estado[df_estado['variable']=='novos']['value'] 
    trace_0 = go.Scatter(
        x=x_values,
        y=y_values, 
        mode='lines',  # 'lines+markers'
        name='Novos',
        marker=marker_dict_0)

    # trace 1
    x_values = df_estado['data'] 
    y_values = df_estado[df_estado['variable']=='acumulado']['value'] 
    trace_1 = go.Scatter(
        x=x_values,
        y=y_values, 
        mode='lines',  # 'lines+markers'
        name='Acumulados',
        marker=marker_dict_1)

    # data is a list of superposed objects in the chart area
    data = [trace_0, trace_1]

    layout = go.Layout(
        title=title,
        xaxis=dict(title=xlabel),
        yaxis=dict(title=ylabel),
        hovermode='closest')
        # updatemenus=updatemenu)

    fig = go.Figure(data=data, layout=layout)
    fig.update_yaxes(type=y_axis_type)

    return fig