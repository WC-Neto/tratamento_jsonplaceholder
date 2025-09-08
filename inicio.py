import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import time

url= "https://jsonplaceholder.typicode.com/comments"

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}

response = requests.get(url, headers=headers)

#print(response.status_code)

texto = response.json() # recebe a requisição via API

df_comentario = pd.DataFrame(texto) #transoforma o json em data frame via pandas

df_comentario['dominios'] = df_comentario['email'].str.split('@').str[1] # cria uma lista de dominios de e-mail

conta_dominios = df_comentario['dominios'].value_counts() #Contagem do número de dominios do maior ao menor

top_dominios = conta_dominios.head(10)# Exibe os 10 primeiros

plt.figure(figsize=(10, 6))

#top_dominios.plot(kind='bar')#cria um gráfico de barras

#plt.show() # plota um gráfico de barras com os 10 dominios mais populares

df_comentario['tamanho_dos_comentarios'] = df_comentario['body'].str.len()


principais_estatisticas = df_comentario['tamanho_dos_comentarios'].describe()

#print(principais_estatisticas) #exibe as principais estatisticas sobre os comentarios como média, max, minimo e afins.

histograma_teste = df_comentario['tamanho_dos_comentarios'].hist()

#plt.figure(figsize=(10, 6))

histograma_teste.plot(kind='bar')

#plt.show() #plotagem do histograma dos comentarios


inicial = time.time()

df_comentario.to_csv('comentarios_analisados.csv', index=False)

final = time.time()

print(f'O tempo de exceução foi de {final-inicial}s')



