import streamlit as st

def main():

    st.title('Extraindo parâmetros epidemiológicos do modelo SIRD a partir dos dados COVID-19 no Brasil')

    st.markdown('''
        Esta aplicação foi criada como um projeto dentro do Bootcamp de _Data Science_ da Alura!  
        **Clique em APP no Menu lateral para explorar!**
        ''')

    st.header('De onde vêm os dados?')
    st.markdown('''
        Os dados utilizados estão vindo direto da fonte, [o GitHub do pesquisador Wesley Cota](https://github.com/wcota/covid19br/). \
        O projeto do Wesley, **covid19br**, usa [informações oficiais do Ministério da Saúde](https://covid.saude.gov.br/) e dados no nível municipal do [Brasil.IO](https://brasil.io/covid19/). \
        Já os dados de recuperados e testes atualmente estão vindo do portal do analista de sitemas [Giscard Stephanou](http://www.giscard.com.br/coronavirus/).
    ''')

    st.header('O que o app faz?')

    st.markdown(r'''
    Faz uma estimativa dos seguintes parâmetros epidemiológicos da epidemia de COVID-19:
    - grau de contágio de uma pessoa infectada ($R_0$), ou seja, quantas pessoas cada pessoa contamina em média quando está infectada,
    - probabilidade de uma pessoa saudável se contaminar ao entra em contato com uma pessoa infectada ($\beta$)
    - taxa diária de recuperação dos pacientes infectados ($\gamma_R$)
    - taxa diária de mortalidade dos pacientes infectados ($\gamma_D$)
    - tempo médio (dias) que os indivíduos permanecem infectados com o virus ($T_{\text{infec}}$).
    ''')

    st.header('Como ele faz?')
    st.markdown(r'''
        Para estimar os parâmetros, considera o modelo de compartilmentos susceptíveis-infectados-recuperados/falecidos (SIRD) [1, 2].  

        O modelo assume que a população saudável $S(t)$ é infectada com uma probabilidade $\beta$ após ter contato com pessoas da população infectada $I(t)$.
        Também é assumido que qualquer pessoa do grupo $S$ (saudáveis) interage com qualquer pessoa do grupo "$I$".
        Após ser infectado, o indivíduo sai do grupo $S$ e migra para o grupo $I$.
        Por fim, os infectados do grupo $I$ podem migrar para o grupo $R$(recuperados), parcela $R(t)$ da população, ou para o grupo $D$ (falecidos, parcela $D(t)$) após um certo tempo médio de infecção.
        A taxa média com que um grupo sai de $I$ para $R$ é $\gamma_R$ (taxa diária de recuperação) e taxa de mortalidade (indivíduo de $I$ migra para $D$) é $\gamma_D$.

        A partir das equações do modelo e dos dados coletados, utilizei a técnica de redução de mínimos quadrados (OLS) para resolver (sucessivamente) os seguintes problemas de otimização:
    ''')
    
    st.latex(r''' 
    \begin{aligned}
        &\min\limits_{\gamma_R} \dfrac{dR(t)}{dt} - \gamma_R I(t) \text{\hspace{18pt} (1)} \\
        &\min\limits_{\gamma_D} \dfrac{dD(t)}{dt} - \gamma_D I(t) \text{\hspace{18pt} (2)} \\
        &\min\limits_{\beta} \dfrac{dI(t)}{dt} + (\gamma_R + \gamma_D) I(t) - \beta \dfrac{S(t)I(t)}{N(t)} \text{\hspace{18pt} (3)}
    \end{aligned}
    ''')
    
    st.markdown(r'''
    Nesses modelos, estou assumindo que o tamanho da população $N(t)$ total não varia, e portanto a relação $N(t) \approx N = S(t) + I(t) + R(t)$ é constante.  
    Embora o modelo seja muito simples, para taxas de mortalidade muito baixas e populações grandes, imagino que o compartimento D não afete tanto a análise.
    ''')

    st.header('DISCLAIMER')
    st.markdown(r'''
                Não sou especialista da área da saúde, e é a primeira vez que tento implementar esses modelos.  
                Ainda tenho um pouco de dificuldade de entender o melhor procedimento a seguir a partir dos dados para estimar os parâmetros.  
                Nas minhas estimativas, considero as seguintes hipóteses:
                - $D(t)$ e $R(t)$ correspondem aos valores acumulados de óbitos e recuperações, extraídos diretamente dos dados coletados;
                - para obter a parcela ativa de infectados $I(t)$, faço a diferença entre o número total de casos (acumulado) e a soma dos números acumulados de recuperados e de óbitos (``I_ativos = I_acc - R - D``);
                - para obter a pacela da população susceptível $S(t)$, faço a diferença entre o tamanho total da população e o número acumulado de casos (``S = N - I_acc``).
                - para qualquer função $x(t)$, estou assumindo que $\dfrac{d x(t)}{dt}$ pode ser aproximado por uma diferença discreta de elementos da série temporal dos dados de $x$ (usando o método ``x.diff(1)`` do ``pandas``).
                - no app, além do estado e do intervalo de datas, você também escolhe o parâmetro do número de dias para a média móvel e percentual para descarte de outliers (altos e baixos), e o procedimento realizado nessa ordem (média móvel $\to$ descarte de outliers), tanto sobre as séries $x(t)$ quanto $dx(t)/dt$.
    ''')

    st.header('Referências Bibliográficas')

    st.markdown(r'''
    - [1] Bastos, S.B., Cajueiro, D.O. [Modeling and forecasting the early evolution of the Covid-19 pandemic in Brazil](https://www.nature.com/articles/s41598-020-76257-1). Sci Rep 10, 19457 (Nov 10, 2020). [DOI 10.1038/s41598-020-76257-1](https://doi.org/10.1038/s41598-020-76257-1)
    - [2] Amaral F, Casaca W, Oishi CM, Cuminato JA. [Towards Providing Effective Data-Driven Responses to Predict the Covid-19 in São Paulo and Brazil](https://www.mdpi.com/1424-8220/21/2/540). Sensors. 2021; 21(2):540. [DOI: 10.3390/s21020540](https://doi.org/10.3390/s21020540)
    - [3] Figueiredo, Flávio. [S(E)IR, COVID-19 e o SUS](https://youtu.be/VtSz59jez-Y). Video aula do Docente da UFMG. 25 de Março de 2020. Disponível no [YouTube](https://youtu.be/VtSz59jez-Y).
    ''')
    
if __name__ == '__main__':
    main()