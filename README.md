# Solução para o Hackathon Forecast Big Data
Este repositório contém a solução desenvolvida por Gustavo Katsuo (Data Prophets) para o Hackathon Forecast Big Data. O objetivo do desafio é construir um modelo de Machine Learning para prever a quantidade de vendas semanais por produto (SKU) e por ponto de venda (PDV) para as cinco primeiras semanas de janeiro de 2023, utilizando o histórico de vendas de 2022 como base.

A métrica oficial de avaliação é o WMAPE (Weighted Mean Absolute Percentage Error).

📈 Estratégia e Metodologia
Nossa estratégia foi centrada em um ciclo iterativo de limpeza de dados, engenharia de features robusta e modelagem, com foco constante na otimização da métrica WMAPE.

1. Limpeza e Tratamento de Dados
O primeiro passo foi consolidar e limpar os três conjuntos de dados fornecidos (transações, produtos e PDVs). As etapas-chave foram:

Integração de Dados: União (merge) das três fontes em um único DataFrame mestre.

Tratamento de Nulos: Foram identificados e tratados os valores nulos resultantes de inconsistências no merge (PDVs presentes nas transações mas não no cadastro).

Detecção e Remoção de Outliers: Através da Análise Exploratória de Dados (EDA), foi identificado um pico de vendas massivo e anômalo em uma única semana de setembro de 2022. Este evento foi considerado um outlier e seus dados foram removidos do conjunto de treino para não contaminar o aprendizado do modelo.

2. Engenharia de Features (Feature Engineering)
Esta foi a etapa mais crítica para a performance do modelo. Foram criadas diversas features para fornecer "pistas" contextuais para o algoritmo:

Features de Tempo: semana_do_ano, dia_da_semana, mes, etc., extraídas da data da transação.

Features de Lag: lag_1_semana, lag_2_semanas, etc., para capturar o volume de vendas das semanas imediatamente anteriores.

Features de Janela Móvel: media_movel_4_semanas e std_movel_4_semanas para capturar a tendência recente e a volatilidade das vendas.

Features Categóricas: Criação de flags binárias para as categorias de produto e PDV mais dominantes (ex. is_package_category), guiado pela análise de Pareto.

Features de Interação: Criação de uma feature poderosa que calcula a média de vendas de cada marca em cada semana_do_ano, capturando a sazonalidade específica da marca.

3. Modelagem
Algoritmo: Foi escolhido o LightGBM (LGBM) Regressor, um modelo de Gradient Boosting conhecido por sua alta velocidade e performance com dados tabulares.

Função Objetivo: Para alinhar o treinamento diretamente com a métrica de avaliação (WMAPE), foi utilizado o objetivo objective='regression_l1' (Mean Absolute Error - MAE), que é um excelente proxy para o WMAPE.

Validação: Foi utilizada uma estratégia de validação temporal, treinando o modelo com os dados até a semana 47 de 2022 e validando sua performance nas semanas restantes do ano.

🛠️ Stack Tecnológico
Linguagem: Python 3.11

Bibliotecas Principais: Pandas, NumPy, Scikit-learn, LightGBM, Matplotlib, Seaborn

Ambiente: Jupyter Notebook executado no VS Code com um ambiente virtual (venv).

🚀 Como Executar o Projeto
Clonar o Repositório:

Bash

git clone https://github.com/gustavokatsuo/Data-Prophets-Big-Data.git
cd Data-Prophets-Big-Data
Criar e Ativar o Ambiente Virtual:

Bash

# No Linux/macOS
python3 -m venv venv
source venv/bin/activate

# No Windows
python -m venv venv
.\venv\Scripts\activate
Instalar as Dependências:

Bash

pip install -r requirements.txt
Adicionar os Dados:

Crie uma pasta chamada data на raiz do projeto.

Coloque os 3 arquivos .parquet fornecidos pelo hackathon dentro desta pasta. (Nota: os dados não estão versionados neste repositório por seu tamanho).

Executar o Notebook:

Abra a pasta do projeto no VS Code.

Abra o notebook principal (ex. analise_e_modelo.ipynb).

Selecione o kernel do venv criado.

Execute as células em ordem. A última célula irá gerar o arquivo submissao_final.csv.

👤 Autor
Gustavo Katsuo