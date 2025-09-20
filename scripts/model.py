# Módulo para treinamento e predição com LightGBM
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import pickle
import os
from .config import MODEL_PARAMS, TRAINING_PARAMS, PREDICTION_WEEKS
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class LightGBMModel:
    """Classe para treinamento e predição com LightGBM"""
    
    def __init__(self, model_params=None, training_params=None):
        self.model_params = model_params or MODEL_PARAMS
        self.training_params = training_params or TRAINING_PARAMS
        self.model = None
        self.feature_names = None
        self.best_iteration = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Treina o modelo LightGBM"""
        print("🤖 Treinando modelo LightGBM...")
        
        # Datasets do LightGBM
        dtrain = lgb.Dataset(X_train, label=y_train)
        valid_sets = [dtrain]
        valid_names = ['training']
        
        if X_val is not None and y_val is not None:
            dval = lgb.Dataset(X_val, label=y_val)
            valid_sets.append(dval)
            valid_names.append('valid_1')
        
        # Treinar modelo
        self.model = lgb.train(
            self.model_params,
            dtrain,
            valid_sets=valid_sets,
            valid_names=valid_names,
            num_boost_round=self.training_params['num_boost_round'],
            callbacks=[
                lgb.early_stopping(self.training_params['early_stopping_rounds']),
                lgb.log_evaluation(self.training_params['verbose_eval'])
            ]
        )
        
        self.feature_names = X_train.columns.tolist()
        self.best_iteration = self.model.best_iteration
        
        # Calcular WMAPE de validação se disponível
        if X_val is not None and y_val is not None:
            pred_val = self.predict(X_val)
            wmape = np.sum(np.abs(pred_val - y_val)) / np.sum(y_val)
            print(f'📊 WMAPE validação: {wmape:.4f}')
            return wmape
        
        return None
    
    def predict(self, X):
        """Faz predições com o modelo treinado"""
        if self.model is None:
            raise ValueError("Modelo não foi treinado. Execute train() primeiro.")
        
        # Usar best_iteration se disponível, senão usar todas as iterações
        if self.best_iteration is not None:
            try:
                return self.model.predict(X, num_iteration=self.best_iteration)
            except TypeError:
                # Para versões mais recentes do LightGBM
                return self.model.predict(X, iteration=self.best_iteration)
        else:
            return self.model.predict(X)
    
    def get_feature_importance(self, importance_type='gain', max_features=None):
        """Retorna feature importance"""
        if self.model is None:
            raise ValueError("Modelo não foi treinado.")
        
        importance = self.model.feature_importance(importance_type=importance_type)
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
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
        
        print(f"💾 Modelo salvo em: {filepath}")
    
    def load_model(self, filepath):
        """Carrega modelo salvo"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.best_iteration = model_data['best_iteration']
        self.model_params = model_data['model_params']
        self.training_params = model_data['training_params']
        
        print(f"📂 Modelo carregado de: {filepath}")
        print(f"   → Treinado em: {model_data.get('timestamp', 'Desconhecido')}")

class Predictor:
    """Classe para gerar predições para novas semanas"""
    
    def __init__(self, model, feature_columns):
        self.model = model
        self.feature_columns = feature_columns
    
    def predict_single_combination(self, pdv, sku, hist_data, weeks=None):
        """Prediz vendas para uma combinação PDV-produto específica"""
        weeks = weeks or PREDICTION_WEEKS
        
        last_series = hist_data['qty'].values.tolist()
        last_gross = hist_data['gross'].values.tolist()
        predictions = []
        
        lags = last_series.copy()
        
        for week in weeks:
            # Construir features
            feat = {
                'week_of_year': week,
                'gross': last_gross[-1] if last_gross else 0
            }
            
            # Lags
            for lag in [1, 2, 3, 4, 8, 12]:
                feat[f'lag_{lag}'] = lags[-lag] if len(lags) >= lag else 0
            
            # Rolling features
            recent_values = lags[-4:] if len(lags) >= 4 else lags
            feat['rmean_4'] = np.mean(recent_values) if recent_values else 0
            feat['rstd_4'] = np.std(recent_values) if len(recent_values) > 1 else 0
            
            nonzero_values = lags[-8:] if len(lags) >= 8 else lags
            feat['nonzero_frac_8'] = np.mean([1 if x > 0 else 0 for x in nonzero_values]) if nonzero_values else 0
            
            # Fazer predição
            X_row = pd.DataFrame([feat])[self.feature_columns]
            pred = max(0, self.model.predict(X_row)[0])
            pred_int = int(round(pred))
            
            predictions.append({
                'semana': week,
                'pdv': int(pdv),
                'produto': int(sku),
                'quantidade': pred_int
            })
            
            # Atualizar lags para próxima iteração
            lags.append(pred)
            if last_gross:
                last_gross.append(last_gross[-1])
        
        return predictions
    
    def predict_all_combinations(self, aggregated_data, weeks=None, parallel=True):
        """Prediz para todas as combinações PDV-produto de forma otimizada"""
        import multiprocessing as mp
        from tqdm import tqdm
        from functools import partial
        
        print("🔮 Gerando predições de forma otimizada...")
        weeks = weeks or PREDICTION_WEEKS
        
        # Obter combinações únicas
        pdv_prod_list = aggregated_data[['pdv', 'internal_product_id']].drop_duplicates().values
        print(f"   → {len(pdv_prod_list):,} combinações PDV-produto para predizer")
        
        # Pré-agrupar dados históricos de forma mais eficiente
        hist_data_map = {}
        for (pdv, sku), group in aggregated_data.groupby(['pdv', 'internal_product_id']):
            hist_data_map[(pdv, sku)] = group.sort_values('week_of_year')
        
        print(f"   → Dados históricos preparados para {len(hist_data_map):,} grupos")
        
        all_predictions = []
        
        if parallel and len(pdv_prod_list) > 100:  # Só usar paralelo se valer a pena
            # Filtrar combinações válidas
            valid_tasks = []
            for pdv, sku in pdv_prod_list:
                hist = hist_data_map.get((pdv, sku))
                if hist is not None and not hist.empty and len(hist) >= 3:  # Mínimo histórico
                    valid_tasks.append((pdv, sku, hist))
            
            print(f"   → {len(valid_tasks):,} combinações válidas para predição")
            
            # Otimização: usar chunksize maior para reduzir overhead
            num_processes = min(mp.cpu_count(), len(valid_tasks))
            chunksize = max(1, len(valid_tasks) // (num_processes * 2))  # Chunks maiores
            
            print(f"   → Utilizando {num_processes} processos com chunksize {chunksize}")
            
            # Criar função para processamento em lote
            def process_chunk(chunk_data):
                chunk_results = []
                for pdv, sku, hist_data in chunk_data:
                    try:
                        predictions = self._predict_single_optimized(pdv, sku, hist_data, weeks)
                        chunk_results.extend(predictions)
                    except Exception as e:
                        # Continuar mesmo se uma predição falhar
                        continue
                return chunk_results
            
            # Dividir em chunks
            chunks = [valid_tasks[i:i + chunksize] for i in range(0, len(valid_tasks), chunksize)]
            
            # Processamento paralelo com chunks
            with mp.Pool(processes=num_processes) as pool:
                results = list(tqdm(
                    pool.imap(process_chunk, chunks), 
                    total=len(chunks),
                    desc="Predições (chunks)"
                ))
            
            # Achatar resultados
            for result_list in results:
                all_predictions.extend(result_list)
                
        else:
            # Processamento sequencial otimizado
            print("   → Usando processamento sequencial otimizado")
            for pdv, sku in tqdm(pdv_prod_list, desc="Predições"):
                hist = hist_data_map.get((pdv, sku))
                if hist is not None and not hist.empty and len(hist) >= 3:
                    try:
                        predictions = self._predict_single_optimized(pdv, sku, hist, weeks)
                        all_predictions.extend(predictions)
                    except Exception:
                        continue
        
        print(f"   → {len(all_predictions):,} predições geradas")
        return pd.DataFrame(all_predictions)
    
    def _predict_single_optimized(self, pdv, sku, hist_data, weeks):
        """Versão otimizada de predição para uma combinação"""
        try:
            predictions = []
            current_data = hist_data.copy()
            
            for week in weeks:
                # Calcular features de forma otimizada
                features = self._calculate_features_fast(current_data)
                
                if features is not None and len(features) == len(self.feature_columns):
                    # Predição usando numpy para velocidade
                    pred = self.model.predict([features], num_iteration=self.model.best_iteration)[0]
                    pred = max(0, pred)  # Não permitir predições negativas
                    
                    predictions.append({
                        'pdv': pdv,
                        'internal_product_id': sku,
                        'week': week,
                        'predicted_qty': pred
                    })
                    
                    # Adicionar predição ao histórico para próxima semana
                    new_row = pd.DataFrame({
                        'week_of_year': [current_data['week_of_year'].max() + 1],
                        'pdv': [pdv],
                        'internal_product_id': [sku],
                        'qty': [pred]
                    })
                    current_data = pd.concat([current_data, new_row], ignore_index=True)
                    
                    # Manter apenas últimas N semanas para eficiência
                    if len(current_data) > 20:
                        current_data = current_data.tail(20)
                        
            return predictions
            
        except Exception as e:
            return []
    
    def _calculate_features_fast(self, data):
        """Calcula features de forma otimizada usando numpy"""
        try:
            if len(data) < 1:
                return None
                
            qty_values = data['qty'].values
            current_week = data['week_of_year'].iloc[-1] + 1
            
            # Features básicas
            features = [current_week]  # week_of_year
            
            # Lags - usar indexação numpy direta
            for lag in [1, 2, 3, 4, 8, 12]:
                if len(qty_values) >= lag:
                    features.append(qty_values[-lag])
                else:
                    features.append(0)
            
            # Rolling means/std - usar numpy para velocidade
            if len(qty_values) >= 4:
                features.append(np.mean(qty_values[-4:]))  # rmean_4
                features.append(np.std(qty_values[-4:]))   # rstd_4
            else:
                features.extend([0, 0])
            
            # Fração de não zeros
            if len(qty_values) >= 8:
                features.append(np.mean(qty_values[-8:] > 0))  # nonzero_frac_8
            else:
                features.append(0)
            
            # Gross (assumir proporcional à quantidade)
            features.append(qty_values[-1] * 10 if len(qty_values) > 0 else 0)
            
            return features
            
        except Exception:
            return None

def train_model(data_dict, save_model_path=None):
    """Função principal para treinar modelo"""
    X_train, y_train = data_dict['train_data']
    X_val, y_val = data_dict['validation_data']
    
    # Inicializar e treinar modelo
    model = LightGBMModel()
    wmape = model.train(X_train, y_train, X_val, y_val)
    
    # Salvar modelo se caminho fornecido
    if save_model_path:
        model.save_model(save_model_path)
    
    return model, wmape

def generate_predictions(model, data_dict, weeks=None, parallel=True, save_path=None):
    """Função principal para gerar predições"""
    predictor = Predictor(model, data_dict['feature_columns'])
    
    predictions_df = predictor.predict_all_combinations(
        data_dict['aggregated_data'], 
        weeks=weeks,
        parallel=parallel
    )
    
    # Salvar predições se caminho fornecido
    if save_path:
        predictions_df.to_csv(save_path, sep=';', index=False, encoding='utf-8')
        print(f"💾 Predições salvas em: {save_path}")
    
    return predictions_df