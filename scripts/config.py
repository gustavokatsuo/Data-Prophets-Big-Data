# Configurações globais do projeto
import os
from multiprocessing import cpu_count

# Caminhos
DATA_PATH = r'data/df_completo_raw.parquet'
OUTPUT_DIR = 'output'
MODELS_DIR = 'models'

# Criar diretórios se não existirem
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Parâmetros do modelo
MODEL_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.05,
    'num_leaves': 64,
    'min_data_in_leaf': 20,
    'verbose': -1,
    'seed': 42,
    'num_threads': cpu_count()
}

# Parâmetros de treinamento
TRAINING_PARAMS = {
    'num_boost_round': 2000,
    'early_stopping_rounds': 100,
    'verbose_eval': 100
}

# Features de lag
LAG_FEATURES = [1, 2, 3, 4, 8, 12]

# Divisão temporal
TRAIN_CUTOFF = 44  # Semanas <= 44 para treino
VALID_START = 45   # Semanas 45-48 para validação
VALID_END = 48

# Predições
PREDICTION_WEEKS = [1, 2, 3, 4, 5]  # Janeiro 2023

# Configurações de visualização
PLOT_CONFIG = {
    'figsize': (18, 12),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'palette': 'husl'
}

# Parâmetros de detecção e tratamento de outliers
OUTLIER_PARAMS = {
    'rolling_window': 12,
    'z_threshold': 2.5,           # Threshold mais agressivo para detectar outliers
    'treatment': 'cap',           # Capping é mais efetivo para outliers extremos
    'adaptive_threshold': True,   # Ajusta threshold baseado nas características do grupo
    'min_observations': 10        # Mínimo de observações para estatísticas confiáveis
}

# Configurações de processamento
PROCESSING_CONFIG = {
    'max_workers': cpu_count(),
    'chunk_size': 1000,
    'progress_update_freq': 1000
}