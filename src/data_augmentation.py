import pandas as pd

def balanced_da(df_annot, metodo="bt"):
    contagem_pares = df_annot.groupby(['category', 'polarity']).size()
    
    teto_absoluto = contagem_pares.max() 
    
    lista_dfs = [df_annot]

    for (cat, pol), total in contagem_pares.items():
        # Definição de n_aug conforme Tabela 3 do material EDA 
        if total <= 500:
            n_aug = 16  # Para datasets muito pequenos 

        # O limite de novas amostras respeita a recomendação técnica
        max_permitido = total * n_aug
        alvo_final = min(teto_absoluto, total + max_permitido)
        diferenca = alvo_final - total
        
        if diferenca > 0:
            df_subset = df_annot[(df_annot['category'] == cat) & (df_annot['polarity'] == pol)]
            df_para_aumentar = df_subset.sample(diferenca, replace=True, random_state=42)
            
            if metodo == "bt":
                df_para_aumentar['text'] = df_para_aumentar['text'].apply(aplicar_bt)
            elif metodo == "sr":
                #EDA sugere alpha=0.1 como ideal para SR
                df_para_aumentar['text'] = df_para_aumentar['text'].apply(aplicar_sr)
                
            lista_dfs.append(df_para_aumentar)
            
    return pd.concat(lista_dfs).reset_index(drop=True)

def gen_all_datasets_combinations(df_annot):
    """Gera as 5 variações solicitadas."""
    datasets = {}
    datasets['original'] = df_annot.copy()
    datasets['bt'] = balanced_da(df_annot, metodo="bt")
    datasets['sr'] = balanced_da(df_annot, metodo="sr")
    
    # Combinações Duplas
    print("Gerando BT + SR")
    datasets['bt_sr'] = balanced_da(datasets['bt'], metodo="sr")
    
    print("Gerando SR + BT")
    datasets['sr_bt'] = balanced_da(datasets['sr'], metodo="bt")
    
    return datasets

def aplicar_bt(text):
    return text
    
    
def aplicar_sr(text):
    return text