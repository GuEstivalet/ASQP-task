<<<<<<< HEAD
import os
import sys
from src.preprocessing.load_and_clean import load_and_clean_dataset
from src.data_analysis.analise_dataset import first_analysis

def main():
    
    # 1. Loading and Cleaning dataset
    try:
        df_raw = load_and_clean_dataset()
    except Exception as e:
            print(f"Critical Error loading data: {e}")
    
    # 2. Visualize and Interpret raw data
    
    first_analysis(df_raw)
    
    # 3. Generate different combinations of dataset (raw, BT, SR, BT+SR, SR+BT)
    # where BT is back translation and SR is Synonym Replacement
    

            

                # B. Balance Data
                df_balanced = balance_group_data(df_block)
                
                # C. Split Train/Test and Sample for Tuning
                X_train, X_test, y_train, y_test, X_train_samp, y_train_samp = split_and_sample(df_balanced)
                
                # C.0 Impute missing values if any
                X_train, X_test, X_train_samp = impute_data(X_train, X_test, X_train_samp)     
                               
                # C.1 Normalize Data if configured
                if ExperimentConfig.NORMALIZE_DATA:
                    X_train, X_test, X_train_samp = normalize_data(X_train, X_test, X_train_samp)
                
                # D. (Optional) Validation / Learning Curves
                if ExperimentConfig.RUN_VALIDATION_CURVES or ExperimentConfig.RUN_LEARNING_CURVES:
                    subdir = f"{grouping_name}_{model_strategie_id}_{group_id_clean}"
                    if ExperimentConfig.RUN_VALIDATION_CURVES:
                        generate_validation_curves(X_train_samp, y_train_samp, subdir, model_type=current_model_type)
                    if ExperimentConfig.RUN_LEARNING_CURVES:
                        generate_learning_curve(X_train_samp, y_train_samp, subdir, model_type=current_model_type, train_sizes=ExperimentConfig.LEARNING_CURVE_TRAIN_SIZES)
                    continue

                # E. Feature Selection (RFE) using dynamic model type
                log_message(f"--- Feature Selection (RFE) ---", level="stage")
                selected_cols = run_rfe(X_train_samp, y_train_samp, current_model_type)
                
                # F. Hyperparameter Tuning (Random Search)
                log_message(f"--- Hyperparameter Tuning ---", level="stage")
                best_params = tune_hyperparameters(X_train_samp[selected_cols], y_train_samp, current_model_type)
                
                # G. Final Training (Full Train set, Selected Features)
                log_message(f"--- Final Training ---", level="stage")
                final_model = train_final_model(X_train[selected_cols], y_train, current_model_type, best_params)
                
                # H. Evaluation (delegated to src.evaluation.evaluate_and_save)
                try:
                    evaluate_and_save(
                        final_model=final_model,
                        X_test=X_test,
                        y_test=y_test,
                        X_train=X_train,
                        y_train=y_train,
                        selected_cols=selected_cols,
                        grouping_name=grouping_name,
                        model_strategie_id=model_strategie_id,
                        block_group=block_group,
                        current_model_type=current_model_type,
                        best_params=best_params,
                        export_model_callback=export_model_to_cpp,
                    )
                except Exception as e:
                    log_message(f"Error during evaluation step: {e}", level="ERROR")
if __name__ == "__main__":
    main()
=======
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
>>>>>>> 69bf944c67e91ccda675686c298dfd1ff646fa2a
