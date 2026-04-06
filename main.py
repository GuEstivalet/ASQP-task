import os
import sys
from src.analise_dataset import carregar_dados,verificar_nulos,processar_anotacoes,analisar_polaridade,analisar_categorias
from src.data_augmentation import balanced_da
def main():
    
    # 1. Setup inicial
    caminho = 'train.json'
    df_raw = carregar_dados(caminho)
    
    # 2. Verificação de integridade
    verificar_nulos(df_raw)
    
    # 3. Processamento e Limpeza
    df_annot = processar_anotacoes(df_raw)
    
    # 4. Análises Estatísticas
    analisar_polaridade(df_annot)
    analisar_categorias(df_annot)
    
    # 5. Geração de Datasets Experimentais
    meus_datasets = balanced_da(df_annot)
    
    
    
    # Teste de como acessar um dataset de meus_datasets
    # df_para_treino = meus_datasets['bt_sr']
    
    # 6. Salvar os datasets 
    """
    Comentado para aplicar futuramente
    for nome, df_exp in meus_datasets.items():
        df_exp.to_csv(f'dataset_{nome}.csv', index=False)
    print("\nDatasets experimentais foram salvos.")
    """
    
if __name__ == "__main__":
    main()
