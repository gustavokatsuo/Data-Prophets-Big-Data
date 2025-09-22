# Módulo para treinamento e predição com XGBoost
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import pickle
import os
from .config import XGBOOST_PARAMS, TRAINING_PARAMS, PREDICTION_WEEKS
from datetime import datetime
import warnings
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
import traceback

warnings.filterwarnings('ignore')

class XGBoostModel:
    """Classe para treinamento e predição com XGBoost"""
    
    def __init__(self, model_params=None, training_params=None):
        self.model_params = model_params or XGBOOST_PARAMS
        self.training_params = training_params or TRAINING_PARAMS
        self.model = None
        self.feature_names = None
        self.best_iteration = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Treina o modelo XGBoost"""
        print("🚀 Treinando modelo XGBoost...")
        
        # Preparar dados
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns.tolist())
        
        # Lista de datasets para monitoramento
        evallist = [(dtrain, 'train')]
        
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=X_train.columns.tolist())
            evallist.append((dval, 'eval'))
        
        # Treinar modelo
        self.model = xgb.train(
            self.model_params,
            dtrain,
            num_boost_round=self.training_params['num_boost_round'],
            evals=evallist,
            early_stopping_rounds=self.training_params['early_stopping_rounds'],
            verbose_eval=self.training_params['verbose_eval']
        )
        
        self.feature_names = X_train.columns.tolist()
        self.best_iteration = self.model.best_iteration
        
        # Calcular WMAPE de validação se disponível
        if X_val is not None and y_val is not None:
            dval_pred = xgb.DMatrix(X_val, feature_names=X_train.columns.tolist())
            pred_val = self.model.predict(dval_pred, iteration_range=(0, self.best_iteration))
            wmape = np.sum(np.abs(pred_val - y_val)) / np.sum(np.abs(y_val))
            print(f'📊 WMAPE validação: {wmape:.4f}')
            return wmape
        
        return None
    
    def predict(self, X):
        """Faz predições com o modelo treinado"""
        if self.model is None:
            raise ValueError("Modelo não foi treinado. Execute train() primeiro.")
        
        # Criar DMatrix para predição
        if isinstance(X, pd.DataFrame):
            dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        else:
            dmatrix = xgb.DMatrix(X)
        
        # Predição usando a melhor iteração
        if self.best_iteration is not None:
            return self.model.predict(dmatrix, iteration_range=(0, self.best_iteration))
        else:
            return self.model.predict(dmatrix)
    
    def get_feature_importance(self, importance_type='gain', max_features=None):
        """Retorna feature importance"""
        if self.model is None:
            raise ValueError("Modelo não foi treinado.")
        
        # XGBoost tem diferentes tipos de importância
        importance_map = self.model.get_score(importance_type=importance_type)
        
        # Converter para DataFrame
        feature_imp = pd.DataFrame(
            list(importance_map.items()), 
            columns=['feature', 'importance']
        ).sort_values('importance', ascending=False)
        
        if max_features:
            feature_imp = feature_imp.head(max_features)
        
        return feature_imp
    
    def save_model(self, filepath):
        """Salva o modelo treinado"""
        if self.model is None:
            raise ValueError("Modelo não foi treinado.")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'best_iteration': self.best_iteration,
            'model_params': self.model_params,
            'training_params': self.training_params,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Modelo XGBoost salvo em: {filepath}")
    
    def load_model(self, filepath):
        """Carrega modelo salvo"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.best_iteration = model_data['best_iteration']
        self.model_params = model_data['model_params']
        self.training_params = model_data['training_params']
        
        print(f"📂 Modelo XGBoost carregado de: {filepath}")
        print(f"   → Treinado em: {model_data.get('timestamp', 'Desconhecido')}")

def _predict_single_xgb(pdv, sku, hist_data, model, feature_columns, weeks):
    """Predição para uma única combinação PDV-produto usando XGBoost"""
    try:
        predictions = []
        qty_values = hist_data['qty'].values
        
        # Extrair features categóricas do último registro
        last_row = hist_data.iloc[-1]
        
        dummy_cols = [col for col in hist_data.columns if col.startswith(('categoria_', 'premise_'))]
        simple_emb_cols = [col for col in hist_data.columns if col.endswith(('_log_freq', '_freq_rank'))]
        
        categorical_features = {}
        if dummy_cols:
            categorical_features.update(last_row[dummy_cols].to_dict())
        if simple_emb_cols:
            categorical_features.update(last_row[simple_emb_cols].to_dict())
        
        # Pre-computar lag indices
        lag_indices = np.array([1, 2, 3, 4, 8, 12])
        
        for week in weeks:
            feat = {'week_of_year': week}
            
            # Features de lag
            n_qty = len(qty_values)
            lag_values = np.zeros(len(lag_indices))
            valid_lags = lag_indices <= n_qty
            if np.any(valid_lags):
                valid_indices = lag_indices[valid_lags]
                lag_values[valid_lags] = qty_values[-valid_indices]
            
            for i, lag in enumerate([1, 2, 3, 4, 8, 12]):
                feat[f'lag_{lag}'] = lag_values[i]
            
            # Rolling features
            if n_qty >= 4:
                recent_values = qty_values[-4:]
                feat['rmean_4'] = np.mean(recent_values)
                feat['rstd_4'] = np.std(recent_values)
            else:
                feat['rmean_4'] = np.mean(qty_values) if n_qty > 0 else 0
                feat['rstd_4'] = np.std(qty_values) if n_qty > 1 else 0
            
            # Nonzero fraction
            if n_qty >= 8:
                nonzero_values = qty_values[-8:]
            else:
                nonzero_values = qty_values
            feat['nonzero_frac_8'] = np.mean(nonzero_values > 0) if len(nonzero_values) > 0 else 0
            
            # Adicionar features categóricas
            feat.update(categorical_features)
            
            # Criar DataFrame para XGBoost
            X_row = pd.DataFrame([feat])
            
            # Garantir que todas as features estão presentes
            for col in feature_columns:
                if col not in X_row.columns:
                    X_row[col] = 0
            
            X_row = X_row[feature_columns]
            
            # Predição com XGBoost
            dmatrix = xgb.DMatrix(X_row, feature_names=feature_columns)
            pred = max(0, model.predict(dmatrix)[0])
            
            predictions.append({
                'pdv': int(pdv),
                'internal_product_id': int(sku),
                'week': week,
                'predicted_qty': pred
            })
            
            # Atualizar histórico
            qty_values = np.append(qty_values, pred)
            if len(qty_values) > 20:
                qty_values = qty_values[-20:]
        
        return predictions
    
    except Exception as e:
        return []

def _predict_batch_xgb(batch_data, model, feature_columns, weeks):
    """Predição em batch para XGBoost"""
    try:
        all_predictions = []
        
        for pdv, sku, hist_data in batch_data:
            predictions = _predict_single_xgb(pdv, sku, hist_data, model, feature_columns, weeks)
            all_predictions.extend(predictions)
        
        return all_predictions
    except Exception as e:
        print(f"--- ERRO em batch XGBoost: {e} ---")
        return []

class XGBoostPredictor:
    """Predictor específico para XGBoost"""
    def __init__(self, model, feature_columns):
        self.model = model
        self.feature_columns = feature_columns
    
    def predict_all_combinations(self, aggregated_data, weeks=None, parallel=True):
        """Prediz para todas as combinações PDV-produto usando XGBoost"""
        print("🚀 Gerando predições com XGBoost...")
        weeks = weeks or PREDICTION_WEEKS
        
        # Filtrar dados inválidos
        print("   → Filtrando dados inválidos...")
        original_count = len(aggregated_data)
        
        aggregated_data_clean = aggregated_data[
            aggregated_data['pdv'].astype(str).str.isdigit()
        ].copy()
        
        aggregated_data_clean['pdv'] = aggregated_data_clean['pdv'].astype(int)
        
        filtered_count = len(aggregated_data_clean)
        if original_count != filtered_count:
            print(f"   → Removidos {original_count - filtered_count:,} registros com PDV inválido")
        
        pdv_prod_list = aggregated_data_clean[['pdv', 'internal_product_id']].drop_duplicates().values
        print(f"   → {len(pdv_prod_list):,} combinações totais encontradas")
        
        # Preparar dados históricos
        print("   → Preparando dados históricos...")
        hist_data_map = {}
        
        grouped_data = aggregated_data_clean.groupby(['pdv', 'internal_product_id'])
        for (pdv, sku), group in grouped_data:
            sorted_group = group.sort_values('week_of_year')
            hist_data_map[(pdv, sku)] = sorted_group
        
        valid_tasks = []
        for pdv, sku in pdv_prod_list:
            hist_data = hist_data_map.get((pdv, sku))
            if hist_data is not None and len(hist_data) >= 3:
                valid_tasks.append((pdv, sku, hist_data))
        
        print(f"   → {len(valid_tasks):,} combinações válidas para predição")
        
        if len(valid_tasks) == 0:
            print("   → ⚠️ Nenhuma combinação válida encontrada.")
            return pd.DataFrame()
        
        all_predictions = []
        
        # Processamento (sem paralelização por enquanto para simplificar)
        print("   → Gerando predições...")
        batch_size = 1000
        for i in tqdm(range(0, len(valid_tasks), batch_size), desc="Predições XGBoost"):
            batch = valid_tasks[i:i + batch_size]
            try:
                predictions = _predict_batch_xgb(batch, self.model.model, self.feature_columns, weeks)
                all_predictions.extend(predictions)
            except Exception:
                continue
        
        print(f"   → {len(all_predictions):,} predições geradas com XGBoost")
        return pd.DataFrame(all_predictions)

def train_xgboost_model(data_dict, save_model_path=None):
    """Função principal para treinar modelo XGBoost"""
    X_train, y_train = data_dict['train_data']
    X_val, y_val = data_dict['validation_data']
    
    # Inicializar e treinar modelo
    model = XGBoostModel()
    wmape = model.train(X_train, y_train, X_val, y_val)
    
    # Salvar modelo se caminho fornecido
    if save_model_path:
        model.save_model(save_model_path)
    
    return model, wmape

def generate_xgboost_predictions(model, data_dict, weeks=None, parallel=True, save_path=None):
    """Função principal para gerar predições com XGBoost"""
    predictor = XGBoostPredictor(model, data_dict['feature_columns'])
    
    predictions_df = predictor.predict_all_combinations(
        data_dict['aggregated_data'], 
        weeks=weeks,
        parallel=parallel
    )
    
    # Reordenar colunas para o formato solicitado
    if len(predictions_df) > 0:
        predictions_formatted = pd.DataFrame({
            'semana': predictions_df['week'].astype(int),
            'pdv': predictions_df['pdv'].astype(int),
            'produto': predictions_df['internal_product_id'].astype(int),
            'quantidade': predictions_df['predicted_qty'].round().astype(int)
        })
    else:
        predictions_formatted = pd.DataFrame(columns=['semana', 'pdv', 'produto', 'quantidade'])
    
    # Salvar predições se caminho fornecido
    if save_path and len(predictions_formatted) > 0:
        predictions_formatted.to_csv(save_path, sep=';', index=False, encoding='utf-8')
        print(f"💾 Predições XGBoost salvas em: {save_path}")
    
    return predictions_formatted