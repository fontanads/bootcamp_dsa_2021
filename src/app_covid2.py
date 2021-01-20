import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

import plotly.offline as pyo
import plotly.graph_objs as go
from plotly import tools

from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

from io import BytesIO
import urllib3

import datetime



def download_url(url, filename):
    http = urllib3.PoolManager() 
    r = http.request('GET', url)
    csv_file = BytesIO(r.data)
    return csv_file

def carrega_e_preprocessa_dataframe(csv_file):
    df = pd.read_csv(csv_file)
    df['date'] = pd.to_datetime(df['date'])
    df['pop']  = ( df['deaths'] / df['deaths_per_100k_inhabitants'] )*100000 # pop em 100k hab
    df['newDeaths_100k'] = df['newDeaths']/( df['pop'] / 100000 ) # 
    
    df['N'] = df['pop']
    df['D'] = df['deaths']
    df['R'] = df['recovered']
    df['I'] = df['totalCases']-df['R']
    df['S'] = df['pop'] - df['I'] - df['R']
    df['IS_div_N'] = df['I']*df['S']/df['N']

    df['state'] = df['state'].replace({'TOTAL':'BRASIL'})
    df = df.set_index('date').sort_index()

    return df

def moving_avg(X,Y,ws):
    Y = Y.rolling(window=ws, center=True).mean().dropna()
    X = X.rolling(window=ws, center=True).mean().dropna()
    return X,Y

def remove_outliers(X,Y,eps):
    q_low = Y.quantile(eps)
    q_hi  = Y.quantile(1-eps)
    filter_locs = (Y < q_hi) & (Y > q_low)

    if sum(filter_locs) > 2:
        Y = Y[filter_locs]
        X = X[filter_locs]
    return X, Y

def ols_reg(X,Y,const):
    if const:
        X= sm.add_constant(X)
    model = OLS(endog=Y, exog=X)
    results = model.fit()
    return results.params, results

def full_ols(df, x_str,y_str, ws=7, eps=0.01, const=True):
    Y = df[y_str]
    X = df[x_str]

    X, Y = moving_avg(X,Y,ws)
    X, Y  = remove_outliers(X,Y,eps)

    params, _ = ols_reg(X,Y,const)

    return (X, Y), params

def get_trace(x_values, y_values, name, color='rgb(54,28,100)', sz=5, lw=1, symbol='hexagon2', mode='lines'):
        marker_dict = dict(
            size=sz,
            color =color,
            symbol=symbol,
            line={'width':lw}
            )
        trace = go.Scatter(
            x=x_values,
            y=y_values, 
            mode=mode,  # 'lines+markers'
            name=name,
            marker=marker_dict)
        return trace

def get_fig_from_traces(data, title, xlabel, ylabel, hovermode='closest'):
        layout = go.Layout(
            title=title,
            xaxis=dict(title=xlabel),
            yaxis=dict(title=ylabel),
            hovermode='closest')
        fig = go.Figure(data=data, layout=layout)
        return fig

def estima_r0_dyn(df, estado, dates_tuple, window_size, eps, window_len_days, stride_val_days):
    const=False
    dynamic_gamma=True
    
    date_begin, date_end   = dates_tuple
    date_begin  = date_begin.strftime('%Y-%m-%d')
    date_end    =   date_end.strftime('%Y-%m-%d')

    # DATASET DO ESTADO E SELEÇÃO DE VARIÁVEIS
    cols = ['N','S','I','R', 'D', 'IS_div_N']
    df_temp = df.query(f'state=="{estado}"')[cols]

    # GERANDO SÉRIES PARA EQUAÇÃO DE DIFERENÇAS SIRD
    df_temp['D_diff'] = df_temp['D'].diff(1)
    df_temp['R_diff'] = df_temp['R'].diff(1)
    df_temp['I_diff'] = df_temp['I'].diff(1)
    df_temp['S_diff'] = df_temp['S'].diff(1)

    # FILTRO NAS DATAS DESEJADAS
    df_temp = df_temp.loc[date_begin:date_end]
    df_temp = df_temp.dropna()

    # PARÂMETRO OLS GAMMA_R
    y_str = 'R_diff'
    x_str = 'I'

    (X, Y), params  = full_ols(df_temp,x_str,y_str, ws=window_size, eps=eps,const=const)
    gamma_r   = params['I']
    Y_gr = Y
    const_gr = params['const'] if const else 0
    ols_eq_gr = gamma_r*X+const_gr

    # traces fig Rdiff = gr*I
    trace0_dRdt = get_trace(Y_gr.index, Y_gr.values, 'dR/dt')
    trace1_dRdt = get_trace(ols_eq_gr.index, ols_eq_gr.values, 'OLS', color ='rgb(150,28,64)')
    
    data = [trace0_dRdt, trace1_dRdt]
    fig1 = get_fig_from_traces(data,
    title=r'Estimando $\gamma_R$ a partir de R_diff e I', 
    xlabel='data', 
    ylabel='dR/dt')
    
    # PARÂMETRO OLS GAMMA_D
    y_str = 'D_diff'
    x_str = 'I'
    (X, Y), params  = full_ols(df_temp, x_str,y_str,ws=window_size, eps=eps,const=const)
    gamma_d   = params['I']
    Y_gd = Y
    const_gd = params['const'] if const else 0
    ols_eq_gd = gamma_d*X + const_gd

    # traces fig Ddiff = gd*I
    trace0_dDdt = get_trace(Y_gd.index, Y_gd.values, 'dD/dt')
    trace1_dDdt = get_trace(ols_eq_gd.index, ols_eq_gd.values, 'OLS', color ='rgb(150,28,64)')
    data = [trace0_dDdt, trace1_dDdt]
    fig2 = get_fig_from_traces(data,
    title=r'Estimando $\gamma_D$ a partir de D_diff e I', 
    xlabel='data', 
    ylabel='dD/dt')

  
    # PARÂMETRO OLS BETA
    df_temp['I_diff_minus_ggI'] = df_temp['I_diff'] + (gamma_r+gamma_d)*df_temp['I']
    y_str = 'I_diff_minus_ggI'
    x_str = 'IS_div_N'

    Y = df_temp[y_str]
    X = df_temp[x_str]

    I_diff    = df_temp['I_diff']
    I_diff, _ = moving_avg(I_diff,I_diff, window_size)
    I_diff, _ = remove_outliers(I_diff,I_diff,eps)

    ggI    = (gamma_r+gamma_d)*df_temp['I']
    ggI, _ = moving_avg(ggI,ggI,window_size)
    ggI, _ = remove_outliers(ggI,ggI,eps)

    (X, Y), params = full_ols(df_temp, x_str,y_str,ws=window_size, eps=eps,const=const)
    Y_b = Y
    beta   = params['IS_div_N']
    const_b  = params['const'] if const else 0
    ols_eq_b = beta*X + const_b

    # traces fig Idiff = beta*SI/N - gg*I
    trace0_dIdt = get_trace(I_diff.index, I_diff.values, r'dI/dt')
    trace1_dIdt = get_trace((ols_eq_b-ggI).index, (ols_eq_b-ggI).values, 'OLS', color ='rgb(150,28,64)')
    data = [trace0_dIdt, trace1_dIdt]
    fig3 = get_fig_from_traces(data,
    title=r'Estimando $\beta$ a partir de I_diff, I, S, N e \gamma (D+R)', 
    xlabel=r'data', 
    ylabel=r'dI/dt')

    # prepare plotly view
    figs_all = tools.make_subplots(rows=3, cols=2,
                              subplot_titles=['I(t)','dI/dt', 'R(t)','dR/dt','D(t)','dD/dt'],
                              shared_xaxes = True,
                              shared_yaxes = False,  # this makes the hours appear only on the left
                              )


    I_curve    = df_temp['I']
    I_curve, _ = moving_avg(I_curve,I_curve, window_size)
    I_curve, _ = remove_outliers(I_curve,I_curve,eps)
    trace_I = get_trace(I_curve.index, I_curve.values, r'I(t)')

    R_curve    = df_temp['R']
    R_curve, _ = moving_avg(R_curve,R_curve, window_size)
    R_curve, _ = remove_outliers(R_curve,R_curve,eps)
    trace_R = get_trace(R_curve.index, R_curve.values, r'R(t)')

    D_curve    = df_temp['D']
    D_curve, _ = moving_avg(D_curve,D_curve, window_size)
    D_curve, _ = remove_outliers(D_curve,D_curve,eps)
    trace_D = get_trace(D_curve.index, D_curve.values, r'D(t)')


    figs_all.append_trace(trace_I, 1, 1)
    figs_all.append_trace(trace0_dIdt, 1, 2)
    figs_all.append_trace(trace1_dIdt, 1, 2)
    
    figs_all.append_trace(trace_R, 2, 1)
    figs_all.append_trace(trace0_dRdt, 2, 2)
    figs_all.append_trace(trace1_dRdt, 2, 2)

    figs_all.append_trace(trace_D, 3, 1)
    figs_all.append_trace(trace0_dDdt, 3, 2)
    figs_all.append_trace(trace1_dDdt, 3, 2)
    
    figs_all['layout'].update(      # access the layout directly!
    title=r'y(t) e dy/dt',
    title_x=0.5
    )

    # figs = (fig1, fig2, fig_I)

    r0t = beta/(gamma_r+gamma_d)
    static_T_infec = 1/(gamma_r+gamma_d)


    df_vars_epi = pd.DataFrame({'vars':[r0t, beta, gamma_r, gamma_d, static_T_infec]}, index=['R0','beta','gamma_r','gamma_d','T_infec'])

    return figs_all, df_vars_epi

def main():
    st.title('Extraindo parâmetros epidemiológicos do modelo SIRD a partir dos dados COVID-19 no Brasil')
    
    # default: '2020-03-01' - '2021-02-01'
    default_dates = [datetime.date(2020, 3, 1), datetime.date(2021, 2, 1)]

    st.sidebar.subheader('Escolha qual estado e janela de tempo deseja analisar:')
    dates_tuple = st.sidebar.date_input(label=r"Escolha o período da janela de estimação dos parâmetros $\beta$ e $\gamma$:", value=default_dates)
    estado_selecionado = st.sidebar.selectbox('Selecione o Estado', todos_os_estados, index=0)

    st.sidebar.subheader('Ajuste alguns parâmetros para pré-processamento dos dados:')
    window_size     = st.sidebar.slider('tamanho da janela de média-móvel:',      min_value=1,             max_value=31, value=1, step=1, format='%d', key=None)
    eps = st.sidebar.slider(r'% dos outliers para descartar:', min_value=0.1, max_value=10., value=0.1, step=0.1, format='%1.3f', key=None) 

    st.sidebar.subheader('Para a curva de R0(t) e Beta(t)/gamma(t):')
    window_len_days = st.sidebar.slider(r'num. de dias da janela de dados:',      min_value=2*window_size, max_value=max(60,2*window_size+1), value=2*window_size, step=1, format='%d', key=None) 
    stride_val_days = st.sidebar.slider(r'dias de avanço para a próxima janela:', min_value=1,             max_value=max(window_len_days,2), value=1, step=1, format='%d', key=None) 

    st.write(r"**Datas Escolhidas**:", tuple([x.strftime(r'%d/%m/%Y') for x in dates_tuple]))
    st.write(r"**Estado atual**:", estado_selecionado)

    
    figs_all, df_vars_epi = estima_r0_dyn(df, estado_selecionado, dates_tuple, window_size, eps/100, window_len_days, stride_val_days)
    # fig1, fig2, fig3 = figs

    # st.dataframe(df_vars_epi.style.highlight_max(axis=0))
    st.dataframe(df_vars_epi)

    # print plots
    st.plotly_chart(figs_all)
    
    # st.latex(r''' \gamma_R = \arg\min\limits_\gamma \dfrac{dR(t)}{dt} - \gamma I(t) ''')
    # st.plotly_chart(fig1)

    # st.latex(r''' \gamma_D = \arg\min\limits_\gamma \dfrac{dD(t)}{dt} - \gamma I(t) ''')
    # st.plotly_chart(fig2)

    # st.latex(r''' \beta = \arg\min\limits_{\beta^\prime} \dfrac{dI(t)}{dt} + (\gamma_R + \gamma_D) I(t) - \beta^\prime \dfrac{S(t)I(t)}{N(t)} ''')
    # st.plotly_chart(fig3)
    

    
if __name__ == '__main__':
    url         = 'https://github.com/wcota/covid19br/raw/master/cases-brazil-states.csv'    
    filename    = 'cases-brazil-states.csv'
    csv_file = download_url(url, filename)    
    df = carrega_e_preprocessa_dataframe(csv_file)
    todos_os_estados = sorted(df['state'].unique().tolist())
    todos_os_estados.remove('BRASIL')
    todos_os_estados.insert(0,'BRASIL')
    main()