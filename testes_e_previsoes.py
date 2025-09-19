import pandas as pd
import numpy as np
import kaggle
from zipfile import ZipFile
import os
from sklearn.model_selection import train_test_split
# Importando as ferramentas necessárias da scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import pandas as pd


print('Entrando no Kaggle')
api =  kaggle.KaggleApi()
api.authenticate()
print('Autenticação realizada com sucesso')

dataset_slug = 'ricardomattos05/jogos-do-campeonato-brasileiro'
nome_do_arquivo_zip = 'database_futebol_brasileiro'
nome_do_arquivo_csv = 'data-raw/csv/brasileirao_matches.csv'


api.dataset_download_files(dataset_slug)

with ZipFile('jogos-do-campeonato-brasileiro.zip', 'r') as nome_do_arquivo_zip:
   nome_do_arquivo_zip.extract(nome_do_arquivo_csv)

#with ZipFile('jogos-do-campeonato-brasileiro.zip', 'r') as arquivo_zip:
    #lista_de_arquivos = arquivo_zip.namelist()
    #print("Ficheiros encontrados dentro do ZIP:")
    #print(lista_de_arquivos) # usei para solucionar um problema de caminho para o arquivo.



dataset = pd.read_csv("data-raw/csv/brasileirao_matches.csv")

#print(f'As 5 primeiras linhas do dataset são compostas por \n {dataset.head()}')
teste = dataset.isna()
#print(f'O maior número de gols feitos por um time da casa é de {dataset.home_goal.max()} gols')
#print(f'O maior número de gols feitos por um time de fora é de {dataset.away_goal.max()} gols')
sem_nulos = dataset.dropna().copy()
sem_nulos['home_goal'] = sem_nulos['home_goal'].astype('int64')
sem_nulos['away_goal'] = sem_nulos['away_goal'].astype('int64')
sem_nulos['datetime'] = pd.to_datetime(sem_nulos['datetime'])
sem_nulos['diasdasemana'] = sem_nulos['datetime'].dt.day_name()
#print(sem_nulos.diasdasemana)
sem_nulos['mes_da_partida'] = sem_nulos['datetime'].dt.month
#print(sem_nulos['mes_da_partida'])
sem_nulos['total_gols'] = sem_nulos['home_goal'] + sem_nulos['away_goal']
media_gols_por_dia = sem_nulos.groupby('diasdasemana')['total_gols'].mean().round(2).sort_values(ascending=False)
#
def definir_resultado(linha):
    gols_casa = linha['home_goal']
    gols_visitante = linha['away_goal']

    if gols_casa > gols_visitante:
        return 'Vitoria_Casa'
    
    elif gols_casa < gols_visitante:
        return 'Vitoria_fora'
    
    else:
        return 'Empate'
    
sem_nulos['resultado'] = sem_nulos.apply(definir_resultado, axis=1)

y = sem_nulos['resultado']

X = sem_nulos.drop(columns='resultado')

dados = {'home_goal': [2, 0, 1, 3], 'away_goal': [1, 0, 2, 3], 'temporada': [2020, 2020, 2021, 2021]}
sem_nulos = pd.DataFrame(dados)
sem_nulos['resultado'] = ['Vitoria_Casa', 'Empate', 'Vitoria_Visitante', 'Empate']
y = sem_nulos['resultado']
X = sem_nulos.drop(columns='resultado')

# \---------------------------------------------------------

print("Dividindo os dados em conjuntos de treino e teste...")

# A função train_test_split recebe as features (X) e o alvo (y)

# e retorna 4 DataFrames/Series.

# test_size=0.2 significa que 20% dos dados irão para o conjunto de teste.

# random_state=42 é como uma "semente" para garantir que a divisão seja a mesma toda vez que rodarmos o código.

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# \--- VERIFICAÇÃO ---

print("\nDivisão concluída com sucesso!")
print(f"Tamanho do conjunto de treino (X_treino): {X_treino.shape}")
print(f"Tamanho do conjunto de teste (X_teste):   {X_teste.shape}")
print(f"Tamanho das respostas de treino (y_treino): {y_treino.shape}")
print(f"Tamanho das respostas de teste (y_teste):   {y_teste.shape}")


X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# \---------------------------------------------------------

print("Preparando os dados para o treinamento...")


colunas_categoricas = ['home_team', 'away_team', 'diasdasemana']



pre_processador = make_column_transformer(
(OneHotEncoder(handle_unknown='ignore'), colunas_categoricas),
remainder='passthrough')

# O OneHotEncoder transforma cada categoria de texto em uma nova coluna de 0s e 1s.

# 'remainder="passthrough"' diz para ele manter as outras colunas (as numéricas) como estão.

# Usamos o RandomForestClassifier. 'n_estimators' é o número de "árvores" na floresta.

# 'random_state=42' garante que o modelo seja o mesmo toda vez que rodarmos.

modelo = RandomForestClassifier(n_estimators=100, random_state=42)

pipeline_completo = make_pipeline(pre_processador, modelo)

print("Iniciando o treinamento do modelo de Machine Learning...")

# O método .fit() é o comando para "aprender".

# Ele usa as "pistas" de treino (X_treino) e as "respostas" de treino (y_treino).

pipeline_completo.fit(X_treino, y_treino)

print("\n--- TREINAMENTO CONCLUÍDO\! ---")