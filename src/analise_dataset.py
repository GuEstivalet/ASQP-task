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
    """
    Transpõe o DF, explode as anotações e normaliza os dicionários 
    em colunas (category, aspect, sentiment, polarity).
    """
    df_transposto = df.T
    # Explodir a lista de anotações para que cada uma ganhe sua própria linha
    df_exploded = df_transposto.explode('annotations')
    # Normalizar os dicionários dentro da coluna annotations
    df_annot = pd.json_normalize(df_exploded['annotations'])
    return df_annot

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
