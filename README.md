# Bootcamp DSA 2021 da Alura Cursos Online

Produções realizadas no [Bootcamp de Data Science Aplicada da Alura](https://www.alura.com.br/bootcamp/data-science-aplicada/matriculas-abertas). Trata-se de uma formação intensiva em Ciência de Dados, utilizando principalmente Python e Jupyter Notebooks.

:health_worker: Na primeira edição, a temática foi **Saúde**, explorando bases de dados abertas de várias fontes:
- [DataSUS Tabnet](https://datasus.saude.gov.br/informacoes-de-saude-tabnet/), 
- [pesquisa de saúde escolar do IBGE](https://www.ibge.gov.br/estatisticas/sociais/educacao/9134-pesquisa-nacional-de-saude-do-escolar.html?=&t=o-que-e), 
- [dados de COVID-19 do governo federal](https://covid.saude.gov.br/), 
- [dados de COVID-19 do hospital Sírilo Libanês](https://www.kaggle.com/S%C3%ADrio-Libanes/covid19), 

dentre outras.

:man_technologist: Algumas _hard-skills_ desenvolvidas no curso:
-  ```pandas``` e
- ```seaborn``` e ```matplotlib```
-  ```statsmodels``` e ```sklearn```
- ```pmdarima``` e ```fbprophet```

:date: Início dia 04/11/2020 e término dia 10/03/2021.

## Módulo 01: Trabalhando com dados do DataSUS

No primeiro módulo tento analisar características importantes dos dados de procedimentos hospitalares. Sazonalidade no número de óbitos em cada ano é um ponto de destaque. Também enumero a natureza das doenças que causam mais óbitos a partir dos dados do SUS entre 2011 e 2018. Doenças respiratórias são a 4ª causa de morte mais frequente, correspondendo a aproximadamente 31% dos óbitos em cada ano.

:green_book: Acesse aqui o meu [Projeto 01](./Bruno_Fontana_da_Silva_M01.ipynb)

## Módulo 02: Dados de Vacinação

:notebook: Projeto em andamento.

## Módulo 03: Dados de Saúde na Educação (PENSE 2015)

Neste módulo, foram analisados os dados [Pesquisa Nacional de Saúde do Escolar - PeNSE](https://www.ibge.gov.br/estatisticas/sociais/educacao/9134-pesquisa-nacional-de-saude-do-escolar.html?=&t=o-que-e). O objetivo foi encontrar pontos importantes através de análise exploratória de dados. Alguns conceitos de testes de normalidade e teste de hipótese (paramétricos e não-paramétricos) foram utilizados para fins de estudo.

:green_book: Acesse aqui o meu [Projeto 03](./Bruno_Fontana_da_Silva_M03.ipynb).

## Módulo 04: Análise de Séries Temporais com dados de COVID-19 no Brasil

Neste módulo, trabalhamos com análise de séries temporais utilizando dados de caso de tuberculose desde 2001, retirados do DataSUS. A ênfase foi na análise de séries temporais, incluindo:
- análise de médias móveis e _resampling_
- funções de autocorrelação
- decomposição de séries temporais (tendência, sazonalidade e resíduos)
- previsões com modelos ARMA, ARIMA, SARIMA e AUTOARIMA.

:notebook: Projeto em andamento.

Optei por dedicar a maior parte do tempo explorando a ferramenta [Streamlit](https://www.streamlit.io/), um framework de desenvolvimento de aplicativos Python voltado para aplicações com ciência de dados e aprendizagem de máquina. A partir do exemplo trabalhado no curso, aprofundei um pouco mais para construir uma aplicação que analisa os dados de COVID-19 e tenta encontrar parâmetros epidemiológicos do modelo matemático SIR (susceptíveis, infectados e recuperados).

:notebook_with_decorative_cover: :iphone: Veja aqui a [página do meu aplicativo no Streamlit](https://share.streamlit.io/fontanads/bootcamp_dsa_2021/main/src/app.py).

:books: Acesse aqui o [código fonte do app](./src/).

## Módulos 05 e 06: Aplicações de Aprendizagem de Máquina na área da Saúde

:notebook: Projeto em andamento.


Em andamento.
