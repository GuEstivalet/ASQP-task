import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def haValorNulo(df):
    print("Há valores nulos?")
    a = df.isnull().any().any()
    print(a)
    return
    

df = pd.read_json('train.json')
haValorNulo(df)


total_reviews = len(df.columns)
print(f"Há um total de {total_reviews} reviews")

# Cada Objeto Json é um review.
# Cada Review é composto por um text e por N annotations
# Cada annotation é composto por category, aspect, sentiment, polarity

# Como está balanceado quanto a polaridade?
df_transposto = df.T 

# 2. "Explodir" a coluna annotations (cada item da lista vira uma linha)
# Isso criará uma linha para cada dicionário de anotação
df_exploded = df_transposto.explode('annotations')

# 3. Transformar os dicionários da coluna annotations em novas colunas
# (category, aspect, sentiment, polarity)
df_annotations = pd.json_normalize(df_exploded['annotations'])

# 4. Contar a polaridade
balanceamento = df_annotations['polarity'].value_counts()
print("Balanceamento por Polaridade:")
print(balanceamento)

# 5. Ver em porcentagem
print("\nEm porcentagem:")
print(df_annotations['polarity'].value_counts(normalize=True) * 100)

''' 
Balanceamento por Polaridade:
polarity
POS    2196
NEG    1179
NEU      61
Name: count, dtype: int64

Em porcentagem:
polarity
POS    63.911525
NEG    34.313155
NEU     1.775320
Name: proportion, dtype: float64
'''

# Problema: dataset desbalanceado - Como tratar?? Core da monografia

# Análise das categorias:

contagem_categorias = df_annotations['category'].value_counts()
porcentagem_categorias = df_annotations['category'].value_counts(normalize=True) * 100


print("--- Distribuição por Categoria ---")
print(contagem_categorias)
print("\n--- Porcentagem por Categoria ---")
print(porcentagem_categorias)

# Reservar uma seção para mostrar se o modelo erra mais a polaridade em categorias que têm menos exemplos.

# Mostra quantas anotações há por combinação de Categoria e Polaridade
print(df_annotations.groupby(['category', 'polarity']).size().unstack(fill_value=0))