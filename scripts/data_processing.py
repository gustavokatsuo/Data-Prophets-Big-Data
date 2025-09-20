# Módulo para tratamento e processamento de dados
import pandas as pd
import numpy as np
from tqdm import tqdm
from config import LAG_FEATURES
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Classe para processamento e engenharia de features dos dados"""
    
    def __init__(self):
        self.df_raw = None
        self.df_agg = None
        self.feature_columns = None
        
    def load_data(self, file_path):
        """Carrega dados do arquivo parquet"""
        print("📊 Carregando dados...")
        self.df_raw = pd.read_parquet(file_path)
        self.df_raw['quantity'] = self.df_raw['quantity'].fillna(0)
        
        # Remover coluna semana_do_ano se existir e criar week_of_year baseado em datas
        self.df_raw = self.df_raw.drop(columns=['semana_do_ano'], errors='ignore')
        min_date = self.df_raw['transaction_date'].min()
        self.df_raw['week_of_year'] = (self.df_raw['transaction_date'] - min_date).dt.days // 7 + 1
        
        print(f"   → Dados carregados: {len(self.df_raw):,} registros")
        return self.df_raw
    
    def aggregate_data(self):
        """Agrega dados por semana/PDV/produto"""
        print("🔄 Agregando dados por semana/PDV/produto...")
        
        # Agregação principal
        self.df_agg = self.df_raw.groupby(['week_of_year', 'pdv', 'internal_product_id'], as_index=False).agg({
            'quantity': 'sum',
            'gross_value': 'sum'
        }).rename(columns={'quantity': 'qty', 'gross_value': 'gross'})
        
        # Garantir continuidade temporal (semanas 1-52)
        all_weeks = pd.DataFrame({'week_of_year': range(1, 53)})
        keys = self.df_agg[['pdv', 'internal_product_id']].drop_duplicates()
        
        # Cross join otimizado
        keys['key'] = 1
        all_weeks['key'] = 1
        grid = keys.merge(all_weeks, on='key').drop('key', axis=1)
        self.df_agg = grid.merge(self.df_agg, on=['pdv', 'internal_product_id', 'week_of_year'], how='left').fillna(0)
        
        print(f"   → Dados agregados: {len(self.df_agg):,} registros")
        print(f"   → PDVs únicos: {self.df_agg['pdv'].nunique():,}")
        print(f"   → Produtos únicos: {self.df_agg['internal_product_id'].nunique():,}")
        
        return self.df_agg
    
    def create_lag_features(self):
        """Cria features de lag e rolling de forma vetorizada"""
        print("⚡ Criando features com operações vetorizadas...")
        
        # Ordenar dados
        self.df_agg = self.df_agg.sort_values(['pdv', 'internal_product_id', 'week_of_year'])
        
        # Criar lags
        print("  → Criando lags...")
        for lag in LAG_FEATURES:
            self.df_agg[f'lag_{lag}'] = self.df_agg.groupby(['pdv', 'internal_product_id'])['qty'].shift(lag)
        
        # Features rolling
        print("  → Criando features rolling...")
        grouped = self.df_agg.groupby(['pdv', 'internal_product_id'])['qty']
        
        self.df_agg['rmean_4'] = grouped.shift(1).rolling(4, min_periods=1).mean()
        print("    → Média móvel 4 criada")
        
        self.df_agg['rstd_4'] = grouped.shift(1).rolling(4, min_periods=1).std()
        print("    → Desvio padrão móvel 4 criado")
        
        # Otimização para fração de não zeros
        is_nonzero = (self.df_agg['qty'] > 0).astype('int8')
        self.df_agg['nonzero_frac_8'] = (
            is_nonzero.groupby([self.df_agg['pdv'], self.df_agg['internal_product_id']])
            .shift(1)
            .rolling(8, min_periods=1)
            .mean()
        )
        print("    → Fração de não zeros nas últimas 8 criada")
        
        # Preencher NAs
        lag_cols = [c for c in self.df_agg.columns if c.startswith('lag_')] + ['rmean_4', 'rstd_4', 'nonzero_frac_8']
        self.df_agg[lag_cols] = self.df_agg[lag_cols].fillna(0)
        
        # Definir colunas de features
        self.feature_columns = ['week_of_year'] + lag_cols + ['gross']
        
        print(f"   → {len(lag_cols)} features criadas")
        return self.df_agg
    
    def split_train_validation(self, train_cutoff=44, valid_start=45, valid_end=48):
        """Divide dados em treino e validação"""
        print("📈 Preparando conjuntos de treino e validação...")
        
        train = self.df_agg[self.df_agg['week_of_year'] <= train_cutoff]
        valid = self.df_agg[(self.df_agg['week_of_year'] >= valid_start) & 
                           (self.df_agg['week_of_year'] <= valid_end)]
        
        print(f"   → Treino: {len(train):,} registros (semanas <= {train_cutoff})")
        print(f"   → Validação: {len(valid):,} registros (semanas {valid_start}-{valid_end})")
        
        return train, valid
    
    def get_basic_stats(self):
        """Retorna estatísticas básicas dos dados"""
        if self.df_agg is None:
            return None
            
        zero_pct = (self.df_agg['qty'] == 0).mean() * 100
        
        stats = {
            'total_records': len(self.df_agg),
            'unique_pdvs': self.df_agg['pdv'].nunique(),
            'unique_products': self.df_agg['internal_product_id'].nunique(),
            'mean_sales': self.df_agg['qty'].mean(),
            'median_sales': self.df_agg['qty'].median(),
            'zero_percentage': zero_pct,
            'mean_gross_value': self.df_agg['gross'].mean(),
            'weeks_range': (self.df_agg['week_of_year'].min(), self.df_agg['week_of_year'].max())
        }
        
        return stats
    
    def prepare_features_target(self, data, target_col='qty'):
        """Prepara features e target para modelagem"""
        if self.feature_columns is None:
            raise ValueError("Features não foram criadas. Execute create_lag_features() primeiro.")
            
        X = data[self.feature_columns]
        y = data[target_col]
        
        return X, y

def process_data(file_path):
    """Função principal para processar dados completos"""
    processor = DataProcessor()
    
    # Pipeline de processamento
    processor.load_data(file_path)
    processor.aggregate_data()
    processor.create_lag_features()
    
    # Dividir dados
    train_data, valid_data = processor.split_train_validation()
    
    # Preparar features e targets
    X_train, y_train = processor.prepare_features_target(train_data)
    X_val, y_val = processor.prepare_features_target(valid_data)
    
    # Estatísticas
    stats = processor.get_basic_stats()
    
    return {
        'processor': processor,
        'raw_data': processor.df_raw,
        'aggregated_data': processor.df_agg,
        'train_data': (X_train, y_train),
        'validation_data': (X_val, y_val),
        'feature_columns': processor.feature_columns,
        'stats': stats
    }