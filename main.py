import sys
from src.preprocessed_data.load_and_clean import load_and_clean_data

def main():
    # 1. Load Data
    try:
        df_raw = load_and_clean_data(general_cfg.FILE_PATH)
        #df_raw = df_raw.sample(n=5000, random_state=42) # TODO: remover depois
    except Exception as e:
        print(f"Error loading data: {e}", level="CRITICAL")
        sys.exit(1)


if __name__ == "__main__":
    main()
