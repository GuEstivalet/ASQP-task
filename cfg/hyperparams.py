from transformers import MT5ForConditionalGeneration, MT5Tokenizer, TrainingArguments
from .settings import ExpParamsCfg

# 1. Configurações Base do Modelo (Arquitetura e Tokenizer)
BASE_MODELS = {
    'mt5_small': {
        'model_name': 'google/mt5-small',
        'class': MT5ForConditionalGeneration,
        'tokenizer': MT5Tokenizer,
        'default_args': {
            'max_source_length': 512,
            'max_target_length': 128,
            'num_beams': 4, # Para inferência/ASQP
            'repetition_penalty': 2.5
        }
    },
    'mt5_base': {
        'model_name': 'google/mt5-base',
        'class': MT5ForConditionalGeneration,
        'tokenizer': MT5Tokenizer,
        'default_args': {
            'max_source_length': 512,
            'max_target_length': 128,
            'num_beams': 4,
            'repetition_penalty': 2.5
        }
    }
}

# 2. Espaço de busca para Hyperparameter Tuning
SEARCH_SPACES = {
    'mt5_small': {
        'learning_rate': [2e-5, 5e-5, 1e-4, 3e-4],
        'per_device_train_batch_size': [4, 8, 16],
        'weight_decay': [0.0, 0.01, 0.1],
        'num_train_epochs': [10, 20, 30],
        'warmup_steps': [0, 100, 500],
        'gradient_accumulation_steps': [1, 2, 4]
    },
    'mt5_base': {
        'learning_rate': [1e-5, 3e-5, 5e-5],
        'per_device_train_batch_size': [2, 4, 8], # Base exige mais VRAM
        'weight_decay': [0.01, 0.1],
        'num_train_epochs': [15, 30],
        'gradient_accumulation_steps': [4, 8]
    }
}

# 3. Parâmetros para Curva de Validação/Monitoramento
VAL_CURVE_PARAMS = {
    'mt5_small': {
        'label_smoothing_factor': [0.0, 0.1, 0.15, 0.2],
        'max_steps': [500, 1000, 2000, 5000], # Avaliar convergência
        'adafactor': [True, False], 
    },
    'mt5_base': {
        'label_smoothing_factor': [0.0, 0.1],
        'learning_rate': [5e-6, 1e-5, 5e-5, 1e-4],
        'weight_decay': [0.001, 0.01, 0.1, 1.0]
    }
}

# 4. Configurações do Trainer (Hugging Face)
# Isso une as configs do seu ExperimentConfig com as necessidades do mT5
def get_training_args(output_dir, model_key, params=None):
    base_params = {
        'output_dir': output_dir,
        'evaluation_strategy': "epoch",
        'save_strategy': "epoch",
        'load_best_model_at_end': True,
        'metric_for_best_model': "eval_loss", # Ou "f1" se tiver o script de extração de quadruplas
        'seed': ExperimentConfig.RANDOM_STATE,
        'push_to_hub': False,
    }
    
    if params:
        base_params.update(params)
        
    return TrainingArguments(**base_params)
