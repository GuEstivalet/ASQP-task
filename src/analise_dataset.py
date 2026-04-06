import json
import pandas as pd
import matplotlib.pyplot as plt

def carregar_dados(caminho_arquivo):
    """Lê o JSON e prepara o DataFrame base."""
    df = pd.read_json(caminho_arquivo)
    print(f"Dados carregados. Total de reviews: {len(df.columns)}")
    return df

def verificar_nulos(df):
    """Verifica se existem valores nulos no DataFrame."""
    tem_nulo = df.isnull().any().any()
    print(f"Há valores nulos? {tem_nulo}")
    return tem_nulo

def processar_anotacoes(df):
    # Transpõe o dataframe
    df_transposto = df.T 
    
    # Cada anotação vira linha
    df_exploded = df_transposto.explode('annotations')
    
    # Reseta o índice para não perder o ID do review e facilitar a concatenação 
    df_exploded = df_exploded.reset_index().rename(columns={'index': 'review_id'})
    
    # Normaliza as anotações
    df_annot_cols = pd.json_normalize(df_exploded['annotations'])
    
    # Concatena o texto original com as colunas normalizadas
    # Agora o df_annot terá: review_id, text, category, aspect, sentiment, polarity
    df_final = pd.concat([
        df_exploded[['review_id', 'text']].reset_index(drop=True),
        df_annot_cols.reset_index(drop=True)
    ], axis=1)
    
    return df_final

def analisar_polaridade(df_annot):
    """Exibe o balanceamento de polaridade em valores e porcentagem."""
    balanceamento = df_annot['polarity'].value_counts()
    porcentagem = df_annot['polarity'].value_counts(normalize=True) * 100
    
    print("\n--- Balanceamento por Polaridade ---")
    print(balanceamento)
    print("\nEm porcentagem:")
    print(porcentagem)
    
    return balanceamento

def analisar_categorias(df_annot):
    """Exibe a distribuição de categorias e o cruzamento com polaridade."""
    contagem = df_annot['category'].value_counts()
    
    print("\n--- Distribuição por Categoria ---")
    print(contagem)
    
    print("\n--- Matriz Categoria x Polaridade ---")
    matriz_cruzada = df_annot.groupby(['category', 'polarity']).size().unstack(fill_value=0)
    print(matriz_cruzada)
    
    return contagem
