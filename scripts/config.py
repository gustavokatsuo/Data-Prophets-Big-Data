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

# Parâmetros do modelo LightGBM
MODEL_PARAMS = {
    'objective': 'tweedie', # Tweedie para dados de contagem com muitos zeros
    'metric': 'mae',
    'learning_rate': 0.02,
    'num_leaves': 32,
    'min_data_in_leaf': 20,
    'verbose': -1, 
    'seed': 42,
    'num_threads': cpu_count()
}

# Parâmetros do modelo XGBoost
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'learning_rate': 0.02,
    'max_depth': 6,
    'min_child_weight': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42,
    'nthread': cpu_count(),
    'silent': 1
}

# Parâmetros de treinamento
TRAINING_PARAMS = {
    'num_boost_round': 2000,
    'early_stopping_rounds': 100,
    'verbose_eval': 100
}

# Features de lag
LAG_FEATURES = [1, 2, 3, 4, 8, 12]

# Features categóricas para dummização
DUMMY_FEATURES = ['categoria', 'premise', 'tipos', 'label']

# Features categóricas para embedding
EMBEDDING_FEATURES = ['subcategoria', 'fabricante', 'descricao', 'marca']

# Configurações de embedding com Autoencoders
EMBEDDING_CONFIG = {
    'n_components': 12,  # Aumentado para melhor representação
    'max_unique_values': 200,  # Aumentado - agora suporta mais categorias eficientemente
    'min_frequency': 10,  # Aumentado para reduzir categorias raras
    'autoencoder_params': {
        'embedding_dim': 12,     # Aumentado para melhor representação
        'epochs': 50,           # Reduzido para treino mais rápido
        'batch_size': 128,      # Aumentado para eficiência
        'learning_rate': 0.002, # Ligeiramente aumentado
        'validation_split': 0.0, # Removido para economizar memória
        'early_stopping_patience': 8,  # Reduzido
        'encoder_layers': [32],         # Arquitetura muito mais leve
        'decoder_layers': [32],         # Arquitetura muito mais leve
        'max_samples_per_category': 100 # Limita amostras por categoria
    }
}

# Divisão temporal
TRAIN_CUTOFF = 44  # Semanas <= 44 para treino
VALID_START = 45   # Semanas 45-52 para validação
VALID_END = 52

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
    'method': 'z_score',                   # Método Z-score com winsorização
    'percentile_cap': 98.0,                # Percentil alto para preservar mais dados
    'absolute_cap': float('inf'),          # Sem cap absoluto para preservar sazonalidade
    'rolling_window': 12,                  # Janela de 12 semanas para estabilidade
    'z_threshold': 4.0,                    # Threshold conservador (4 desvios padrão)
    'treatment': 'winsorize',              # WINSORIZAÇÃO - preserva distribuição
    'adaptive_threshold': True,            # Ajuste adaptativo por características do grupo
    'min_observations': 8,                 # Mínimo de observações para cálculos confiáveis
    'absolute_cap_multiplier': 5.0         # Cap como múltiplo da média (backup)
}

# Configurações de processamento
PROCESSING_CONFIG = {
    'max_workers': cpu_count(),
    'chunk_size': 1000,
    'progress_update_freq': 1000
}