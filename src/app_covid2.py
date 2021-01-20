import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

from io import BytesIO
import urllib3

import datetime
from dateutil.relativedelta import relativedelta


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
    df['I'] = df['totalCases']-df['R'] - df['D']
    df['S'] = df['pop'] - df['totalCases'] #- df['I'] - df['R'] - df['D']
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
    X, Y = remove_outliers(X,Y,eps)

    params, _ = ols_reg(X,Y,const)

    return (X, Y), params

def ols_estimative(param_name, df,x_str,y_str, window_size,eps, const):
        (X, Y), params  = full_ols(df, x_str, y_str, ws=window_size, eps=eps,const=const)
        a = params[param_name]
        a = max(1e-6, a) # todos os parâmetros que estão sendo estimados são positivos ou zero
        b = params['const'] if const else 0
        Yest = a*X+b
        return Y, Yest, a

def get_trace(x_values, y_values, name, color='rgb(54,28,100)', sz=5, lw=1, symbol='hexagon2', mode='lines',visible=True):
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
            marker=marker_dict,
            visible=visible)
        return trace

def get_fig_from_traces(data, title, xlabel, ylabel, hovermode='closest'):
        layout = go.Layout(
            title=title,
            xaxis=dict(title=xlabel),
            yaxis=dict(title=ylabel),
            hovermode='closest',
            )
        fig = go.Figure(data=data, layout=layout)
        return fig

def generate_basic_trace_fixed_params(df,col,window_size,eps,label='col'):
        X    = df[col]
        X, _ = moving_avg(X,X, window_size)
        X, _ = remove_outliers(X,X,eps)
        trace = get_trace(X.index, X.values, label)
        return trace

def estima_r0_dyn(df, estado, dates_tuple, window_size, eps):
    const=True

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

    Y_gr, Y_gr_est, gamma_r = ols_estimative(param_name=x_str, df=df_temp,x_str=x_str,y_str=y_str, window_size=window_size,eps=eps, const=const)

    # traces fig Rdiff = gr*I
    trace0_dRdt = get_trace(Y_gr.index, Y_gr.values, 'dR/dt', visible='legendonly')
    trace1_dRdt = get_trace(Y_gr_est.index, Y_gr_est.values, 'OLS', color ='rgb(150,28,64)')
    
    # PARÂMETRO OLS GAMMA_D
    y_str = 'D_diff'
    x_str = 'I'

    Y_gd, Y_gd_est, gamma_d = ols_estimative(param_name=x_str, df=df_temp,x_str=x_str,y_str=y_str, window_size=window_size,eps=eps, const=const)

    # traces fig Ddiff = gd*I
    trace0_dDdt = get_trace(Y_gd.index, Y_gd.values, 'dD/dt', visible='legendonly')
    trace1_dDdt = get_trace(Y_gd_est.index, Y_gd_est.values, 'OLS', color ='rgb(150,28,64)')
   
    # PARÂMETRO OLS BETA
    df_temp['ggI']             = (gamma_r+gamma_d)*df_temp['I']
    df_temp['I_diff_plus_ggI'] = df_temp['I_diff'] + df_temp['ggI']
    
    y_str = 'I_diff_plus_ggI'
    x_str = 'IS_div_N'

    ggI_X        = df_temp['ggI']
    ggI_Y        = df_temp[y_str]
    ggI_X, ggI_Y = moving_avg(ggI_X, ggI_Y,window_size)
    ggI_X, ggI_Y = remove_outliers(ggI_X,ggI_Y,eps)

    # I_diff = df_temp['I_diff']
    # ggI_Y        = df_temp[y_str]
    # I_diff, ggI_Y = moving_avg(I_diff, ggI_Y,window_size)
    # I_diff, ggI_Y = remove_outliers(I_diff,ggI_Y,eps)

    Y_b, Y_b_est, beta = ols_estimative(param_name=x_str, df=df_temp,x_str=x_str,y_str=y_str, window_size=window_size,eps=eps, const=const)

    # traces fig Idiff = beta*SI/N - gg*I
    # trace0_dIdt = get_trace(   (Y_b).index,         (Y_b).values, r'beta*SI/N', visible='legendonly') # r'dI/dt - gamma_est*I' = beta*SI/N
    # trace1_dIdt = get_trace((Y_b_est).index, (Y_b_est).values, 'OLS', color ='rgb(150,28,64)')

    # trace0_dIdt = get_trace(   (ggI_Y-ggI_X).index,         (ggI_Y-ggI_X).values, r'dI/dt')
    # trace1_dIdt = get_trace((Y_b_est-ggI_X).index, (Y_b_est-ggI_X).values, 'OLS', color ='rgb(150,28,64)')

    trace0_dIdt = get_trace((Y_b-ggI_X).index, (Y_b-ggI_X).values, r'dI/dt', visible='legendonly')
    trace1_dIdt = get_trace((Y_b_est-ggI_X).index, (Y_b_est-ggI_X).values, 'OLS', color ='rgb(150,28,64)')

    # prepare plotly view
    figs_all = make_subplots(rows=3, cols=2,
                              subplot_titles=['I(t)','dI/dt', 'R(t)','dR/dt','D(t)','dD/dt'],
                              shared_xaxes = True,
                              shared_yaxes = False,  # this makes the hours appear only on the left
                              )


    trace_I, trace_R, trace_D = [generate_basic_trace_fixed_params(df_temp, col, window_size, eps, label=col+r'(t)') for col in [r'I',r'R',r'D']]


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

    # DATAFRAME COM VALORES DOS PARÂMETROS ESTIMADOS
    r0t = beta/(gamma_r+gamma_d)
    static_T_infec = 1/(gamma_r+gamma_d)
    gammas_static = (gamma_r, gamma_d)
    df_vars_epi = pd.DataFrame({'vars':[r0t, beta, gamma_r, gamma_d, static_T_infec]}, index=['R0','beta','gamma_r','gamma_d','T_infec'])

    return figs_all, df_vars_epi, (df_temp, static_T_infec, gammas_static)

def dynamic_r0(df, dates_tuple, window_size, eps, window_len_days, stride_val_days, static_T_infec, gammas_static, dynamic_gamma=True):
    const = True
    date_begin, date_end   = dates_tuple
    betas = []
    r0s = []
    r0s_static_gamma = []
    T_infec_rds = []

    while date_begin < (date_end - relativedelta(days=window_len_days)):
        date_begin = (date_begin + relativedelta(days=stride_val_days))
        date_end_ts   = (date_begin + relativedelta(days=window_len_days))

        # date_begin_str  = date_begin.strftime('%Y-%m-%d')
        # date_end_str    = date_end_ts.strftime('%Y-%m-%d')  

        cols = ['N','S','I','R','D','IS_div_N','I_diff_plus_ggI','R_diff','D_diff']
        df_temp = df.loc[date_begin:date_end_ts][cols]
        df_temp = df_temp.dropna()

        if len(df_temp)<window_size:
            continue

        if dynamic_gamma:
            x_str = 'I'
            y_str = 'R_diff'
            _, _, gamma_r = ols_estimative(param_name=x_str, df=df_temp,x_str=x_str,y_str=y_str, window_size=window_size,eps=eps, const=const)
            y_str  = 'D_diff'
            _, _, gamma_d = ols_estimative(param_name=x_str, df=df_temp,x_str=x_str,y_str=y_str, window_size=window_size,eps=eps, const=const)
        else:
            gamma_r, gamma_d = gammas_static

        y_str = 'I_diff_plus_ggI'
        x_str = 'IS_div_N'
        _, _, beta = ols_estimative(param_name=x_str, df=df_temp,x_str=x_str,y_str=y_str, window_size=window_size,eps=eps, const=const)
        
        # print('DEBUG BETA')
        # # print(type(beta))
        # print(beta.shape)
        # print(beta)

        dt_offset = (date_end - date_begin)/2
        date_avg = date_end - dt_offset
        # CALCULA R0(t) para GAMMA ESTÁTICO e GAMMA(t)
        r0t_static_gamma =  beta*static_T_infec
        r0t = beta/(gamma_r+gamma_d)

        # APPENDS
        T_infec_rds.append((date_avg,1/(gamma_r+gamma_d)))
        betas.append((date_avg,beta))
        r0s.append((date_avg,r0t))
        r0s_static_gamma.append((date_avg,r0t_static_gamma))
    # FIM WHILE
    
    # prepare plotly view
    figs_all = make_subplots(rows=3, cols=1,
                              subplot_titles=['Beta(t)','R0(t)','Tinfec'],
                              shared_xaxes = True,
                              shared_yaxes = False,  # this makes the hours appear only on the left
                              )

    x, y  = list(zip(*betas))
    trace_beta = get_trace(x, y, name=r'$\beta(t)$') 
    
    x, y  = list(zip(*r0s))
    trace_r0t = get_trace(x, y, name=r'$r0(t)=B(t)/g(t)$') 
    
    x, y  = list(zip(*r0s_static_gamma))
    trace_r0s = get_trace(x, y, name=r'$r0(t)=B(t)/g$',color='rgb(150,28,64)') 

    x, y  = list(zip(*T_infec_rds))
    trace_T_infec_rds = get_trace(x, y, name=r'T_infec(t)') 

    figs_all.append_trace(trace_beta, 1, 1)

    figs_all.append_trace(trace_r0t, 2, 1)
    figs_all.append_trace(trace_r0s, 2, 1)
    figs_all.add_shape(type='line',y0=1,y1=1, x0=x[0], x1=x[-1], line=dict(color='rgb(0,0,0)'), xref='x2',yref='y2')
    figs_all.update_yaxes(range=[.2, 4.0], row=2, col=1)

    figs_all.append_trace(trace_T_infec_rds, 3, 1)
    figs_all.add_shape(type='line',y0=static_T_infec,y1=static_T_infec, x0=x[0], x1=x[-1], line=dict(color='rgb(0,0,0)'), xref='x3',yref='y3')
    figs_all.update_yaxes(range=[0, min(60,max(y)+5)], row=3, col=1)


    figs_all['layout'].update(      # access the layout directly!
    title=r'R0(t) e Gamma(t)',
    title_x=0.5
    )
    return figs_all


# ---------------------------------------------------------------------------------------
#                     MAIN FUNCTION
# ---------------------------------------------------------------------------------------
def main():
    df, todos_os_estados = carrega_dados_cache()

    # st.title('Extraindo parâmetros epidemiológicos do modelo SIRD a partir dos dados COVID-19 no Brasil')
    
    # default: '2020-03-01' - '2021-02-01'
    #           2020/10/01
    default_dates = [datetime.date(2020, 7, 1), datetime.date(2021, 2, 1)]

    st.sidebar.subheader('Escolha qual estado e janela de tempo deseja analisar:')
    dates_tuple = st.sidebar.date_input(label=r"Escolha o período da janela de estimação dos parâmetros $\beta$ e $\gamma$:", value=default_dates)
    estado_selecionado = st.sidebar.selectbox('Selecione o Estado', todos_os_estados, index=0)

    st.sidebar.subheader('Ajuste alguns parâmetros para pré-processamento dos dados:')
    window_size     = st.sidebar.slider('tamanho da janela de média-móvel:',      
                                         min_value=1, max_value=31, value=7, step=1, format='%d', key=None)
    eps = st.sidebar.slider(r'% dos outliers para descartar:',
                            min_value=0.01, max_value=10., value=0.01, step=0.01, format='%1.3f', key=None) 

    st.sidebar.subheader('Para a curva de R0(t) e Beta(t)/gamma(t):')
    window_len_days = st.sidebar.slider(r'J: num. de dias da janela de dados:',
                                          min_value=2*window_size, max_value=max(60,2*window_size+1), value=max(60,2*window_size+1), step=1, format='%d', key=None) 
    stride_val_days = st.sidebar.slider(r'stride: dias de avanço para a próxima janela:', 
                                          min_value=1, max_value=max(window_len_days,2),  value=1, step=1, format='%d', key=None) 
    dynamic_gamma = st.sidebar.radio("Assumir que Gamma também varia dependendo da janela?", [True, False], index=0)

    st.header(r"**Estado atual**:"+estado_selecionado)
    st.write(r"**Datas Escolhidas**:", tuple([x.strftime(r'%d/%m/%Y') for x in dates_tuple]))
   

    
    figs_all, df_vars_epi, (df_temp, static_T_infec, gammas_static) = estima_r0_dyn(df, estado_selecionado, dates_tuple, window_size, eps/100)
    figs_R0t = dynamic_r0(df_temp, dates_tuple, window_size, eps/100, window_len_days, stride_val_days, static_T_infec, gammas_static, dynamic_gamma=dynamic_gamma)
    

    # st.dataframe(df_vars_epi.style.highlight_max(axis=0))
    col1, col2 = st.beta_columns(2)

    with st.beta_container():
        with col1:
            st.subheader(r"Parâmetros Estimados considerando todos os dados entre a data inicial e a data final")
            st.dataframe(df_vars_epi)

        with col2:
            st.subheader(r"Visualização das séries temporais do modelo SIRD usadas nas estimações:")
            st.plotly_chart(figs_all)
        
        st.subheader(r'Notas:')
        st.markdown(r''' 
                    - $I(t)$ e $dI/dt$* são os infectados ativos e a taxa de crescimento dos infectados ativos
                    - $R(t)$ e $dR/dt$ são os recuperados e a taxa de crescimento dos recuperados
                    - $D(t)$ e $dD/dt$ são os óbitos e a taxa de crescimento dos óbitos
                    - as séries originais $dx/dt$ estão desligadas por padrão, para que você visualize o reultado do modelo que as aproxima
                    - ainda preciso melhora a estimativa de $\beta$, pois como você pode notar há um "atraso" na troca de sinal de $dI/dt$ em relação
                    ao momento que $I(t)$ realmente começa a decrescer. Além disso, como $S/N$ (proporção de susceptíveis) varia muito pouco na janela,
                    a curva $dI/dt$ realmente parece ficar proporcional à $I(t)$ por um fator $\beta \times (S/N - \gamma)$.
                    Acredito que isso melhora se usar uma estimação a partir da integral das séries, conforme os artigos sugerem.
                    ''')
    # print plots
    with st.beta_container():
        st.subheader(r"Aplicando a estimativa com janela uma deslizante:")
        st.markdown(r'''
                     Começando a partir da data de início, deslocando-se a cada ``stride`` dias e estimando $\beta(t)$ e $\gamma(t)$ numa janela de ``J`` dias para frente.  

                     Aqui as curas são bem sensíveis aos parâmetros do menu lateral, pois se você diminui muito a janela para estimar as variáveis epidemiológicas, ao deslocar para as próximas janelas a mudança pode ser muito brusca.
                     ''')
        st.plotly_chart(figs_R0t)
        st.markdown(r'''
                     Lembrando a interpretação dos parâmetros epidemiológicos do modelo SIR(D):
                     - $\beta(t)$ é o grau de contágio, a chance de uma pessoa saudável se contaminar ao entrar em contato com as pessoas infectadas
                     - $R_0(t)$ é a medida de quantas pessoas são contaminadas diariamente por uma pessoa infectada (a linha em 1 está destacada pois idealmente 
                     a pandemia acabaria espontaneamente, ao longo do tempo, se conseguíssemos manter $R<1$ tempo suficiente)
                     - $T_{\text{infec}}$ é o tempo médio que uma pessoa infectada permanece transmitindo o vírus antes de se recuperar (também podemos pensar na relação que isso tem no sistema de saúde regional,
                     por isso mantive a linha estática indicando esse valor estimado com a janela inteira, a fim de comparar com a janela deslizante; 
                     mudando o último marcador para ``False`` você elimina a variação de $\gamma$, a taxa de recuperação, na estimativa de $\beta$)
                     - a dependência no tempo acontece por tentar estimar o parâmetro em janelas "mais curtas" em relação ao tamanho completo do dataset (pode occorrer algum erro ao selecionar uma faixa de datas menor que a J),
                     considerando que na história da série houve muita mudança no cenário de medidas não-farmacêuticas de contenção da epidemia, no país e nas regiões.                     
                     ''')

    
@st.cache
def carrega_dados_cache(url='https://github.com/wcota/covid19br/raw/master/cases-brazil-states.csv',filename='cases-brazil-states.csv'):
    csv_file = download_url(url, filename)    
    df = carrega_e_preprocessa_dataframe(csv_file)
    todos_os_estados = sorted(df['state'].unique().tolist())
    todos_os_estados.remove('BRASIL')
    todos_os_estados.insert(0,'BRASIL')
    return df, todos_os_estados

if __name__ == '__main__':
    main()