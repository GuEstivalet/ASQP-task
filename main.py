import os
import sys
from src.analise_dataset import carregar_dados,verificar_nulos,processar_anotacoes,analisar_polaridade,analisar_categorias
from src.data_augmentation import gerar_datasets_experimentais
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
    # Esta função retorna um dicionário com todos os DataFrames
    meus_datasets = gerar_datasets_experimentais(df_annot)
    
    # Exemplo de como acessar um dataset específico:
    # df_para_treino = meus_datasets['bt_sr']
    
    # 6. Salvar os datasets (opcional, mas recomendado para não reprocessar)
    for nome, df_exp in meus_datasets.items():
        df_exp.to_csv(f'dataset_{nome}.csv', index=False)
        
    print("\nTodos os datasets experimentais foram salvos e estão prontos para o treinamento.")
if __name__ == "__main__":
    main()
