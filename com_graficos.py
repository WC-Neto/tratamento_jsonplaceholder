import pandas as pd
import kaggle
import zipfile
import os
import matplotlib.pyplot as plt
import numpy as np
import requests 

def coletar_e_preparar_dados(dataset_slug, pasta_destino, nome_do_csv_no_zip):
    """
    Função principal para coletar dados da API do Kaggle, extrair,
    limpar e fazer a engenharia de features. Retorna um DataFrame pronto.
    """
    try:
        # --- 1. COLETA E PREPARAÇÃO DOS DADOS ---
        print('Entrando no Kaggle...')
        api = kaggle.KaggleApi()
        api.authenticate()
        print('Autenticação realizada com sucesso.')

        print(f'Baixando o dataset: {dataset_slug}')
        api.dataset_download_files(dataset_slug, path=pasta_destino, unzip=True, quiet=False)
        print('Download e extração concluídos.')

        caminho_completo_csv = f"{pasta_destino}/{nome_do_csv_no_zip}"
        dataset = pd.read_csv(caminho_completo_csv)
        print('Dados carregados com sucesso.')

        # --- 2. LIMPEZA E ENGENHARIA DE FEATURES ---
        print('Iniciando limpeza e engenharia de features...')
        sem_nulos = dataset.dropna(subset=['home_goal', 'away_goal']).copy()
        sem_nulos['home_goal'] = sem_nulos['home_goal'].astype('int64')
        sem_nulos['away_goal'] = sem_nulos['away_goal'].astype('int64')
        sem_nulos['datetime'] = pd.to_datetime(sem_nulos['datetime'])
        sem_nulos['diasdasemana'] = sem_nulos['datetime'].dt.day_name()
        sem_nulos['total_gols'] = sem_nulos['home_goal'] + sem_nulos['away_goal']

        def definir_resultado(linha):
            if linha['home_goal'] > linha['away_goal']:
                return 'Vitoria_Casa'
            elif linha['home_goal'] < linha['away_goal']:
                return 'Vitoria_Visitante'
            else:
                return 'Empate'
        sem_nulos['resultado'] = sem_nulos.apply(definir_resultado, axis=1)
        print('Engenharia de features concluída.')
        return sem_nulos

    except Exception as e:
        # Tratamento de erro genérico para a coleta de dados
        print(f"\n--- ERRO CRÍTICO ---")
        print(f"Não foi possível coletar ou preparar os dados. Verifique sua conexão ou as credenciais da API do Kaggle.")
        print(f"Detalhes do erro: {e}")
        return None

def gerar_visualizacoes(df):
    """
    Recebe um DataFrame limpo e gera todas as visualizações.
    """
    if df is None or df.empty:
        print("Não há dados para gerar visualizações.")
        return

    print('\nGerando visualizações...')

    # --- Gráfico 1: Média de Gols por Dia da Semana ---
    plt.figure(figsize=(10, 6))
    media_gols_dia = df.groupby('diasdasemana')['total_gols'].mean().sort_values(ascending=False)
    media_gols_dia.plot(kind='bar', color='skyblue')
    plt.title('Média de Gols por Dia da Semana (Brasileirão)', fontsize=16)
    plt.xlabel('Dia da Semana', fontsize=12)
    plt.ylabel('Média de Gols por Partida', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # --- Gráfico 2: Proporção de Resultados ---
    plt.figure(figsize=(8, 8))
    contagem_resultado = df['resultado'].value_counts()
    contagem_resultado.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'gold', 'lightgreen'])
    plt.title('Proporção de Resultados das Partidas', fontsize=16)
    plt.ylabel('')
    plt.axis('equal')

    # --- Gráfico 3: Média de Gols por Temporada ---
    plt.figure(figsize=(12, 6))
    df['season'] = df['datetime'].dt.year
    media_gols_temporada = df.groupby('season')['total_gols'].mean()
    media_gols_temporada.plot(kind='line', marker='o', linestyle='-')
    plt.title('Evolução da Média de Gols por Temporada', fontsize=16)
    plt.xlabel('Temporada', fontsize=12)
    plt.ylabel('Média de Gols por Partida', fontsize=12)
    plt.grid(True)
    plt.xticks(media_gols_temporada.index.astype(int))
    plt.tight_layout()
    
    # --- Exibição Final ---
    print("Exibindo gráficos...")
    plt.show()

# --- BLOCO DE EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    SLUG_DATASET = 'ricardomattos05/jogos-do-campeonato-brasileiro'
    PASTA_DADOS = 'dados_brasileirao'
    CSV_NO_ZIP = 'data-raw/csv/brasileirao_matches.csv'

    # 1. Tenta coletar e processar os dados
    dataframe_final = coletar_e_preparar_dados(SLUG_DATASET, PASTA_DADOS, CSV_NO_ZIP)

    # 2. Se a primeira etapa foi bem-sucedida, gera os gráficos
    if dataframe_final is not None:
        gerar_visualizacoes(dataframe_final)
        print("\nAnálise visual concluída.")
    else:
        print("\nO programa será encerrado devido a uma falha na coleta de dados.")

