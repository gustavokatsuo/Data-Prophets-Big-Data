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

def _clean_feature_names(columns):
    """Limpa nomes de features para compatibilidade com XGBoost"""
    cleaned = []
    seen = set()
    
    for col in columns:
        # Substitui caracteres problemáticos
        cleaned_name = str(col).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_')
        # Remove outros caracteres especiais que podem causar problemas
        cleaned_name = cleaned_name.replace('%', 'pct').replace('(', '_').replace(')', '_')
        cleaned_name = cleaned_name.replace(' ', '_').replace('-', '_').replace('/', '_')
        
        # Remove múltiplos underscores consecutivos
        while '__' in cleaned_name:
            cleaned_name = cleaned_name.replace('__', '_')
        # Remove underscores no início e fim
        cleaned_name = cleaned_name.strip('_')
        
        # Garantir que não há duplicatas
        original_cleaned = cleaned_name
        counter = 1
        while cleaned_name in seen:
            cleaned_name = f"{original_cleaned}_{counter}"
            counter += 1
            
        seen.add(cleaned_name)
        cleaned.append(cleaned_name)
    
    return cleaned

class XGBoostModel:
    """Classe para treinamento e predição com XGBoost"""
    
    def __init__(self, model_params=None, training_params=None):
        self.model_params = model_params or XGBOOST_PARAMS
        self.training_params = training_params or TRAINING_PARAMS
        self.model = None
        self.feature_names = None
        self.cleaned_feature_names = None
        self.best_iteration = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Treina o modelo XGBoost com suporte nativo para categóricas"""
        print("🚀 Treinando modelo XGBoost...")
        
        # Limpar nomes das features para compatibilidade com XGBoost
        self.feature_names = X_train.columns.tolist()
        self.cleaned_feature_names = _clean_feature_names(self.feature_names)
        
        # Otimização: Garantir que colunas categóricas estão com o tipo correto para XGBoost
        # O pipeline de data_processing já deve fazer isso, mas é uma boa prática garantir.
        for col in X_train.select_dtypes(include=['category']).columns:
            X_train[col] = X_train[col].cat.codes
            if X_val is not None:
                X_val[col] = X_val[col].cat.codes

        # Preparar dados com nomes limpos
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.cleaned_feature_names, enable_categorical=True)
        
        evallist = [(dtrain, 'train')]
        
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.cleaned_feature_names, enable_categorical=True)
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
        
        self.best_iteration = self.model.best_iteration
        
        # Calcular WMAPE de validação se disponível
        if X_val is not None and y_val is not None:
            pred_val = self.predict(X_val)
            wmape = np.sum(np.abs(pred_val - y_val)) / np.sum(np.abs(y_val))
            print(f'📊 WMAPE validação: {wmape:.4f}')
            return wmape
        
        return None
    
    def predict(self, X):
        """Faz predições com o modelo treinado"""
        if self.model is None:
            raise ValueError("Modelo não foi treinado. Execute train() primeiro.")
        
        # Otimização: Converte para DMatrix uma única vez
        # Garante que as colunas categóricas sejam tratadas como códigos se ainda não foram
        X_pred = X.copy()
        for col in X_pred.select_dtypes(include=['category']).columns:
            X_pred[col] = X_pred[col].cat.codes
            
        dmatrix = xgb.DMatrix(X_pred, feature_names=self.cleaned_feature_names, enable_categorical=True)
        
        # Predição usando a melhor iteração
        if self.best_iteration is not None:
            return self.model.predict(dmatrix, iteration_range=(0, self.best_iteration))
        else:
            return self.model.predict(dmatrix)
    
    def get_feature_importance(self, importance_type='gain', max_features=None):
        """Retorna feature importance"""
        if self.model is None:
            raise ValueError("Modelo não foi treinado.")
        
        importance_map = self.model.get_score(importance_type=importance_type)
        
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
        
        # O Booster do XGBoost pode ser salvo diretamente
        self.model.save_model(filepath.replace('.pkl', '.json'))

        # Salvar metadados separadamente
        metadata = {
            'feature_names': self.feature_names,
            'cleaned_feature_names': self.cleaned_feature_names,
            'best_iteration': self.best_iteration,
            'model_params': self.model_params,
            'training_params': self.training_params,
            'timestamp': datetime.now().isoformat()
        }
        with open(filepath.replace('.pkl', '_meta.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"💾 Modelo XGBoost salvo em: {filepath.replace('.pkl', '.json')}")
    
    def load_model(self, filepath):
        """Carrega modelo salvo"""
        self.model = xgb.Booster()
        self.model.load_model(filepath.replace('.pkl', '.json'))
        
        with open(filepath.replace('.pkl', '_meta.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        self.feature_names = metadata['feature_names']
        self.cleaned_feature_names = metadata.get('cleaned_feature_names', _clean_feature_names(self.feature_names))
        self.best_iteration = metadata['best_iteration']
        self.model_params = metadata['model_params']
        self.training_params = metadata['training_params']
        
        print(f"📂 Modelo XGBoost carregado de: {filepath.replace('.pkl', '.json')}")
        print(f"   → Treinado em: {metadata.get('timestamp', 'Desconhecido')}")

def _predict_single_fast_xgb(pdv, sku, hist_data, model, feature_columns, cleaned_feature_names, weeks, categorical_features):
    """Versão ultra-rápida de predição para XGBoost com features em cache."""
    try:
        predictions = []
        qty_values = hist_data['qty'].values
        
        lag_indices = np.array([1, 2, 3, 4, 8, 12])
        
        for week in weeks:
            features_dict = {'week_of_year': week}
            
            # Lags vetorizados
            n_qty = len(qty_values)
            lag_values = np.zeros(6)
            valid_mask = lag_indices <= n_qty
            if np.any(valid_mask):
                lag_values[valid_mask] = qty_values[-lag_indices[valid_mask]]
            
            for i, lag in enumerate([1, 2, 3, 4, 8, 12]):
                features_dict[f'lag_{lag}'] = lag_values[i]
            
            # Rolling features com correção de data leakage
            hist_para_media = qty_values[:-1] if n_qty > 1 else np.array([])
            if len(hist_para_media) >= 4:
                window = hist_para_media[-4:]
                features_dict['rmean_4'] = np.mean(window)
                features_dict['rstd_4'] = np.std(window)
            else:
                features_dict['rmean_4'] = np.mean(hist_para_media) if len(hist_para_media) > 0 else 0
                features_dict['rstd_4'] = np.std(hist_para_media) if len(hist_para_media) > 1 else 0

            # Nonzero fraction
            hist_para_nonzero = qty_values[:-1] if n_qty > 1 else np.array([])
            nz_window = hist_para_nonzero[-8:] if len(hist_para_nonzero) >= 8 else hist_para_nonzero
            features_dict['nonzero_frac_8'] = np.mean(nz_window > 0) if len(nz_window) > 0 else 0
            
            features_dict.update(categorical_features)
            
            # Criar DataFrame para XGBoost com uma linha
            X_row = pd.DataFrame([features_dict])
            X_row = X_row[feature_columns] # Garantir a ordem das colunas
            
            for col in X_row.select_dtypes(include=['category']).columns:
                X_row[col] = X_row[col].cat.codes

            dmatrix = xgb.DMatrix(X_row, feature_names=cleaned_feature_names, enable_categorical=True)
            pred = max(0, model.predict(dmatrix)[0])
            
            predictions.append({
                'pdv': int(pdv),
                'internal_product_id': int(sku),
                'week': week,
                'predicted_qty': pred
            })
            
            qty_values = np.append(qty_values, pred)
            if len(qty_values) > 20:
                qty_values = qty_values[-20:]
        
        return predictions
    
    except Exception:
        return []

def _predict_batch_optimized_xgb(batch_data, model, feature_columns, cleaned_feature_names, weeks):
    """Predição em batch otimizada com cache de features para XGBoost."""
    try:
        all_predictions = []
        categorical_cache = {}
        
        for pdv, sku, hist_data in batch_data:
            cache_key = (pdv, sku)
            if cache_key not in categorical_cache:
                last_row = hist_data.iloc[-1]
                dummy_cols = [col for col in hist_data.columns if col.startswith(('categoria_', 'premise_'))]
                simple_emb_cols = [col for col in hist_data.columns if col.endswith(('_log_freq', '_freq_rank'))]
                
                categorical_features = {}
                if dummy_cols:
                    categorical_features.update(last_row[dummy_cols].to_dict())
                if simple_emb_cols:
                    categorical_features.update(last_row[simple_emb_cols].to_dict())
                categorical_cache[cache_key] = categorical_features
            
            categorical_features = categorical_cache[cache_key]
            
            predictions = _predict_single_fast_xgb(pdv, sku, hist_data, model, feature_columns, cleaned_feature_names, weeks, categorical_features)
            all_predictions.extend(predictions)
        
        return all_predictions
    except Exception as e:
        print(f"--- ERRO em batch XGBoost: {e} ---")
        traceback.print_exc()
        return []

def _process_prediction_chunk_xgb(chunk_args):
    """Wrapper para processar chunks de predições em paralelo para XGBoost."""
    chunk_data, model, feature_columns, cleaned_feature_names, weeks = chunk_args
    return _predict_batch_optimized_xgb(chunk_data, model, feature_columns, cleaned_feature_names, weeks)


class XGBoostPredictor:
    """Predictor otimizado e paralelizado para XGBoost."""
    def __init__(self, model, feature_columns):
        self.model = model
        self.feature_columns = feature_columns
        self.cleaned_feature_names = model.cleaned_feature_names
    
    def predict_all_combinations(self, aggregated_data, weeks=None, parallel=True):
        """Prediz para todas as combinações PDV-produto usando XGBoost de forma otimizada."""
        print("🚀 Gerando predições com XGBoost de forma otimizada...")
        weeks = weeks or PREDICTION_WEEKS
        
        print("   → Filtrando dados inválidos...")
        aggregated_data_clean = aggregated_data[aggregated_data['pdv'].astype(str).str.isdigit()].copy()
        aggregated_data_clean['pdv'] = aggregated_data_clean['pdv'].astype(int)
        
        pdv_prod_list = aggregated_data_clean[['pdv', 'internal_product_id']].drop_duplicates().values
        print(f"   → {len(pdv_prod_list):,} combinações totais encontradas")
        
        print("   → Preparando dados históricos...")
        hist_data_map = {
            (pdv, sku): group.sort_values('week_of_year') 
            for (pdv, sku), group in aggregated_data_clean.groupby(['pdv', 'internal_product_id'])
        }
        
        valid_tasks = [
            (pdv, sku, hist_data_map[(pdv, sku)]) 
            for pdv, sku in pdv_prod_list 
            if (pdv, sku) in hist_data_map and len(hist_data_map[(pdv, sku)]) >= 3
        ]
        
        print(f"   → {len(valid_tasks):,} combinações válidas para predição")
        
        if not valid_tasks:
            print("   → ⚠️ Nenhuma combinação válida encontrada.")
            return pd.DataFrame()
            
        # Para datasets muito grandes, usar versão ultra-otimizada
        if len(valid_tasks) > 100000:
            print("   → Dataset muito grande detectado. Usando versão ultra-otimizada...")
            return self._predict_ultra_fast(valid_tasks, weeks)
            
        all_predictions = []

        if parallel and len(valid_tasks) > 500:
            num_processes = min(mp.cpu_count(), 8) # Limitar a 8 cores para não sobrecarregar
            chunksize = max(500, len(valid_tasks) // (num_processes * 4))
            print(f"   → Utilizando {num_processes} processos com chunksize {chunksize}...")
            
            chunks = [valid_tasks[i:i + chunksize] for i in range(0, len(valid_tasks), chunksize)]
            chunk_args = [(chunk, self.model.model, self.feature_columns, self.cleaned_feature_names, weeks) for chunk in chunks]
            
            with mp.Pool(processes=num_processes) as pool:
                results = list(tqdm(pool.imap(_process_prediction_chunk_xgb, chunk_args), total=len(chunks), desc="Predições XGBoost (Paralelo)"))
            
            for result_list in results:
                all_predictions.extend(result_list)
        else:
            print("   → Usando processamento sequencial otimizado...")
            batch_size = 2000
            for i in tqdm(range(0, len(valid_tasks), batch_size), desc="Predições XGBoost"):
                batch = valid_tasks[i:i + batch_size]
                predictions = _predict_batch_optimized_xgb(batch, self.model.model, self.feature_columns, self.cleaned_feature_names, weeks)
                all_predictions.extend(predictions)
        
        print(f"   → {len(all_predictions):,} predições geradas com XGBoost")
        return pd.DataFrame(all_predictions)

    def _predict_ultra_fast(self, valid_tasks, weeks):
        """Versão ultra-rápida para datasets muito grandes usando processamento em lotes massivos"""
        print("   → Executando predição ultra-rápida...")
        all_predictions = []
        
        # Processar em mega-lotes para máxima eficiência
        mega_batch_size = 5000
        total_batches = (len(valid_tasks) + mega_batch_size - 1) // mega_batch_size
        
        for i in tqdm(range(0, len(valid_tasks), mega_batch_size), 
                     desc="Mega-batches", total=total_batches):
            mega_batch = valid_tasks[i:i + mega_batch_size]
            try:
                # Usar a função de batch otimizada
                batch_predictions = _predict_batch_optimized_xgb(mega_batch, self.model.model, self.feature_columns, self.cleaned_feature_names, weeks)
                all_predictions.extend(batch_predictions)
            except Exception as e:
                print(f"   → Erro no mega-batch {i//mega_batch_size + 1}: {e}")
                continue
        
        print(f"   → {len(all_predictions):,} predições geradas (ultra-rápido)")
        return pd.DataFrame(all_predictions)

def train_xgboost_model(data_dict, save_model_path=None):
    """Função principal para treinar modelo XGBoost."""
    X_train, y_train = data_dict['train_data']
    X_val, y_val = data_dict['validation_data']
    
    # Otimização: Converter colunas de objeto para categoria antes de treinar
    for col in X_train.select_dtypes(include=['object']).columns:
        X_train[col] = X_train[col].astype('category')
        if X_val is not None:
            X_val[col] = X_val[col].astype('category')

    model = XGBoostModel()
    wmape = model.train(X_train, y_train, X_val, y_val)
    
    if save_model_path:
        model.save_model(save_model_path)
    
    return model, wmape

def generate_xgboost_predictions(model, data_dict, weeks=None, parallel=True, save_path=None):
    """Função principal para gerar predições com XGBoost de forma robusta."""
    predictor = XGBoostPredictor(model, data_dict['feature_columns'])
    
    predictions_df = predictor.predict_all_combinations(
        data_dict['aggregated_data'], 
        weeks=weeks,
        parallel=parallel
    )
    
    if not predictions_df.empty:
        predictions_formatted = pd.DataFrame({
            'semana': predictions_df['week'].astype(int),
            'pdv': predictions_df['pdv'].astype(int),
            'produto': predictions_df['internal_product_id'].astype(int),
            'quantidade': predictions_df['predicted_qty'].round().astype(int)
        })
    else:
        predictions_formatted = pd.DataFrame(columns=['semana', 'pdv', 'produto', 'quantidade'])
    
    if save_path and not predictions_formatted.empty:
        predictions_formatted.to_csv(save_path, sep=';', index=False, encoding='utf-8')
        print(f"💾 Predições XGBoost salvas em: {save_path}")
    
    return predictions_formatted