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
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
import traceback # Importar para depuração

warnings.filterwarnings('ignore')

def _predict_batch_optimized(batch_data, model, feature_columns, weeks):
    """Predição em batch otimizada para múltiplas combinações"""
    try:
        all_predictions = []
        
        # Extrair features categóricas uma vez por combinação PDV-SKU
        categorical_cache = {}
        
        for pdv, sku, hist_data in batch_data:
            # Cache de features categóricas por combinação
            cache_key = (pdv, sku)
            if cache_key not in categorical_cache:
                last_row = hist_data.iloc[-1]
                dummy_cols = [col for col in hist_data.columns if col.startswith(('categoria_', 'premise_'))]
                emb_cols = [col for col in hist_data.columns if col.endswith(('_target_enc', '_emb_0', '_emb_1', '_emb_2', '_emb_3', '_emb_4', 
                                   '_emb_5', '_emb_6', '_emb_7', '_emb_8', '_emb_9'))]
                
                categorical_features = {}
                if dummy_cols:
                    categorical_features.update(last_row[dummy_cols].to_dict())
                if emb_cols:
                    categorical_features.update(last_row[emb_cols].to_dict())
                categorical_cache[cache_key] = categorical_features
            
            # Usar features categóricas do cache
            categorical_features = categorical_cache[cache_key]
            
            # Predição rápida para esta combinação
            predictions = _predict_single_fast(pdv, sku, hist_data, model, feature_columns, weeks, categorical_features)
            all_predictions.extend(predictions)
        
        return all_predictions
    except Exception as e:
        print(f"--- ERRO em batch: {e} ---")
        return []

def _predict_single_fast(pdv, sku, hist_data, model, feature_columns, weeks, categorical_features):
    """Versão ultra-rápida de predição com features categóricas em cache"""
    try:
        predictions = []
        qty_values = hist_data['qty'].values
        
        # Pre-computar valores constantes
        lag_indices = np.array([1, 2, 3, 4, 8, 12])
        
        for week in weeks:
            # Preparar features base de forma vetorizada
            features_dict = {'week_of_year': week}
            
            # Lags vetorizados
            n_qty = len(qty_values)
            lag_values = np.zeros(6)  # Para os 6 lags
            valid_mask = lag_indices <= n_qty
            if np.any(valid_mask):
                lag_values[valid_mask] = qty_values[-lag_indices[valid_mask]]
            
            # Adicionar lags ao dict
            for i, lag in enumerate([1, 2, 3, 4, 8, 12]):
                features_dict[f'lag_{lag}'] = lag_values[i]
            
            # Rolling features otimizadas
            if n_qty >= 4:
                window = qty_values[-4:]
                features_dict['rmean_4'] = np.mean(window)
                features_dict['rstd_4'] = np.std(window)
            else:
                features_dict['rmean_4'] = np.mean(qty_values) if n_qty > 0 else 0
                features_dict['rstd_4'] = np.std(qty_values) if n_qty > 1 else 0
            
            # Nonzero fraction
            nz_window = qty_values[-8:] if n_qty >= 8 else qty_values
            features_dict['nonzero_frac_8'] = np.mean(nz_window > 0) if len(nz_window) > 0 else 0
            
            # Adicionar features categóricas do cache
            features_dict.update(categorical_features)
            
            # Criar array de features na ordem correta
            feature_array = np.zeros(len(feature_columns))
            for i, col in enumerate(feature_columns):
                feature_array[i] = features_dict.get(col, 0)
            
            # Predição
            pred = max(0, model.predict(feature_array.reshape(1, -1))[0])
            
            predictions.append({
                'pdv': pdv,
                'internal_product_id': sku,
                'week': week,
                'predicted_qty': pred
            })
            
            # Atualizar histórico para próxima semana
            qty_values = np.append(qty_values, pred)
            if len(qty_values) > 20:  # Manter apenas últimas 20 semanas
                qty_values = qty_values[-20:]
        
        return predictions
    except Exception as e:
        return []

def _process_prediction_chunk(chunk_args):
    """Função auxiliar para processar chunks de predições em paralelo (versão otimizada)"""
    chunk_data, model, feature_columns, weeks = chunk_args
    # Usar a nova função de batch otimizada
    return _predict_batch_optimized(chunk_data, model, feature_columns, weeks)

def _predict_single_optimized_static(pdv, sku, hist_data, model, feature_columns, weeks):
    """Versão estática otimizada de predição para uma combinação"""
    try:
        predictions = []
        current_data = hist_data.copy()
        
        for week in weeks:
            features = _calculate_features_fast_static(current_data)
            
            if features is not None and len(features) == len(feature_columns):
                pred = model.predict(np.array([features]))[0]
                pred = max(0, pred)
                
                predictions.append({
                    'pdv': pdv,
                    'internal_product_id': sku,
                    'week': week,
                    'predicted_qty': pred
                })
                
                new_row = pd.DataFrame({
                    'week_of_year': [current_data['week_of_year'].max() + 1],
                    'pdv': [pdv], 'internal_product_id': [sku], 'qty': [pred]
                })
                
                # Copiar features categóricas do último registro
                last_row = current_data.iloc[-1]
                for col in current_data.columns:
                    if (col.startswith(('categoria_', 'premise_')) or 
                        col.endswith(('_target_enc', '_emb_0', '_emb_1', '_emb_2', '_emb_3', '_emb_4', 
                                     '_emb_5', '_emb_6', '_emb_7', '_emb_8', '_emb_9')) or
                        col in ['categoria', 'premise', 'subcategoria', 'tipos', 'label', 'fabricante']):
                        if col not in new_row.columns:
                            new_row[col] = last_row[col]
                
                current_data = pd.concat([current_data, new_row], ignore_index=True)
                
                if len(current_data) > 20:
                    current_data = current_data.tail(20)
        return predictions
    
    except Exception as e:
        print(f"--- ERRO ao prever para PDV {pdv}, SKU {sku}: {e} ---")
        traceback.print_exc()
        return []
    
def _calculate_features_fast_static(data):
    """Calcula features de forma otimizada usando numpy (versão estática)"""
    try:
        if len(data) < 1:
            return None
            
        qty_values = data['qty'].values
        current_week = data['week_of_year'].iloc[-1] + 1
        
        features = [current_week]
        
        # Lags otimizados com numpy indexing
        lag_indices = np.array([1, 2, 3, 4, 8, 12])
        n_vals = len(qty_values)
        lag_values = np.zeros(len(lag_indices))
        
        # Usar indexing vetorizado para lags válidos
        valid_lags = lag_indices <= n_vals
        if np.any(valid_lags):
            valid_indices = lag_indices[valid_lags]
            lag_values[valid_lags] = qty_values[-valid_indices]
        
        features.extend(lag_values.tolist())

        # Dados para rolling (todos exceto o mais recente) - otimizado
        rolling_data = qty_values[:-1] if len(qty_values) > 1 else qty_values

        # Rolling means/std sobre os últimos 4 pontos *do passado* - vetorizado
        if len(rolling_data) >= 4:
            window = rolling_data[-4:]
            features.extend([np.mean(window), np.std(window)])  # rmean_4, rstd_4
        else: # Se não houver dados suficientes, usar a média/std de tudo que tiver
            if len(rolling_data) > 0:
                mean_val = np.mean(rolling_data)
                std_val = np.std(rolling_data) if len(rolling_data) > 1 else 0.0
                features.extend([mean_val, std_val])
            else:
                features.extend([0.0, 0.0])
        
        # Fração de não zeros sobre os últimos 8 pontos *do passado* - vetorizado
        if len(rolling_data) >= 8:
            features.append(np.mean(rolling_data[-8:] > 0))  # nonzero_frac_8
        else:
            features.append(np.mean(rolling_data > 0) if len(rolling_data) > 0 else 0.0)
        
        features.append(qty_values[-1] * 10 if len(qty_values) > 0 else 0)
        
        # Adicionar features categóricas do último registro (vetorizado)
        # Estas features são constantes para cada combinação PDV-produto
        last_row = data.iloc[-1]
        
        # Features dummy - usar mask vetorizada
        dummy_cols = [col for col in data.columns if col.startswith(('categoria_', 'premise_'))]
        features.extend(last_row[dummy_cols].values if dummy_cols else [])
        
        # Features de embedding - usar mask vetorizada
        emb_cols = [col for col in data.columns if col.endswith(('_target_enc', '_emb_0', '_emb_1', '_emb_2', '_emb_3', '_emb_4', 
                           '_emb_5', '_emb_6', '_emb_7', '_emb_8', '_emb_9'))]
        features.extend(last_row[emb_cols].values if emb_cols else [])
        
        return features
        
    except Exception as e:
        # Também podemos adicionar depuração aqui se necessário
        print(f"--- ERRO em _calculate_features_fast_static: {e} ---")
        return None
    
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
            wmape = np.sum(np.abs(pred_val - y_val)) / np.sum(np.abs(y_val))
            print(f'📊 WMAPE validação: {wmape:.4f}')
            return wmape
        
        return None
    
    def predict(self, X):
        """Faz predições com o modelo treinado, lidando com a mudança de versão do LightGBM."""
        if self.model is None:
            raise ValueError("Modelo não foi treinado. Execute train() primeiro.")
        
        if self.best_iteration is None:
            return self.model.predict(X)

        # Use num_iteration instead of iteration (deprecated in newer LightGBM versions)
        try:
            return self.model.predict(X, num_iteration=self.best_iteration)
        except (TypeError, ValueError):
            try:
                return self.model.predict(X, iteration=self.best_iteration)
            except:
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
    def __init__(self, model, feature_columns):
        self.model = model
        self.feature_columns = feature_columns
    
    def predict_single_combination(self, pdv, sku, hist_data, weeks=None):
        """Prediz vendas para uma combinação PDV-produto específica"""
        weeks = weeks or PREDICTION_WEEKS
        predictions = []
        
        # VALIDAR PDV ANTES DO PROCESSAMENTO
        try:
            # Garantir que PDV seja numérico
            if isinstance(pdv, str) and not pdv.isdigit():
                print(f"⚠️ PDV inválido encontrado: {pdv}. Pulando...")
                return []
            
            pdv = int(pdv)  # Converter para inteiro
            sku = int(sku)  # Garantir que SKU também seja inteiro
        except (ValueError, TypeError) as e:
            print(f"⚠️ Erro ao converter PDV/SKU: pdv={pdv}, sku={sku}. Erro: {e}")
            return []
        
        lags = hist_data['qty'].values  # Manter como numpy array
        
        # Extrair features categóricas do último registro (são constantes para cada PDV-produto)
        last_row = hist_data.iloc[-1]
        
        # Otimizar extração de features categóricas usando masks
        dummy_cols = [col for col in hist_data.columns if col.startswith(('categoria_', 'premise_'))]
        emb_cols = [col for col in hist_data.columns if col.endswith(('_target_enc', '_emb_0', '_emb_1', '_emb_2', '_emb_3', '_emb_4', 
                           '_emb_5', '_emb_6', '_emb_7', '_emb_8', '_emb_9'))]
        
        categorical_features = {}
        if dummy_cols:
            categorical_features.update(last_row[dummy_cols].to_dict())
        if emb_cols:
            categorical_features.update(last_row[emb_cols].to_dict())
        
        # Pre-computar lag indices para otimização
        lag_indices = np.array([1, 2, 3, 4, 8, 12])
        
        for week in weeks:
            feat = {'week_of_year': week}
            
            # Features de lag otimizadas com numpy indexing
            n_lags = len(lags)
            lag_values = np.zeros(len(lag_indices))
            valid_lags = lag_indices <= n_lags
            if np.any(valid_lags):
                valid_indices = lag_indices[valid_lags]
                lag_values[valid_lags] = lags[-valid_indices]
            
            # Adicionar lags ao dict de features
            for i, lag in enumerate([1, 2, 3, 4, 8, 12]):
                feat[f'lag_{lag}'] = lag_values[i]
            
            # Features rolling otimizadas
            if n_lags >= 4:
                recent_values = lags[-4:]
                feat['rmean_4'] = np.mean(recent_values)
                feat['rstd_4'] = np.std(recent_values)
            else:
                feat['rmean_4'] = np.mean(lags) if n_lags > 0 else 0
                feat['rstd_4'] = np.std(lags) if n_lags > 1 else 0
            
            # Nonzero fraction otimizada
            if n_lags >= 8:
                nonzero_values = lags[-8:]
            else:
                nonzero_values = lags
            feat['nonzero_frac_8'] = np.mean(nonzero_values > 0) if len(nonzero_values) > 0 else 0
            
            # Adicionar features categóricas
            feat.update(categorical_features)
            
            # Criar DataFrame e selecionar apenas as features que o modelo conhece
            X_row = pd.DataFrame([feat])
            
            # Garantir que todas as features do modelo estão presentes
            for col in self.feature_columns:
                if col not in X_row.columns:
                    X_row[col] = 0
            
            X_row = X_row[self.feature_columns]
            pred = max(0, self.model.predict(X_row)[0])
            
            # Padronização dos nomes das colunas de saída para evitar inconsistências.
            predictions.append({
                'pdv': int(pdv),
                'internal_product_id': int(sku),
                'week': week,
                'predicted_qty': pred  # Usar o valor float, consistente com a outra função
            })
            
            lags = np.append(lags, pred)  # Usar numpy append para arrays
        
        return predictions
    
    def predict_all_combinations(self, aggregated_data, weeks=None, parallel=True):
        """Prediz para todas as combinações PDV-produto de forma otimizada"""
        print("🔮 Gerando predições de forma otimizada...")
        weeks = weeks or PREDICTION_WEEKS
        
        # FILTRAR PDVs INVÁLIDOS (como 'desconhecido') ANTES DO PROCESSAMENTO
        print("   → Filtrando dados inválidos...")
        original_count = len(aggregated_data)
        
        # Remover registros com PDV não numérico
        aggregated_data_clean = aggregated_data[
            aggregated_data['pdv'].astype(str).str.isdigit()
        ].copy()
        
        # Converter PDV para inteiro para garantir consistência
        aggregated_data_clean['pdv'] = aggregated_data_clean['pdv'].astype(int)
        
        filtered_count = len(aggregated_data_clean)
        if original_count != filtered_count:
            print(f"   → Removidos {original_count - filtered_count:,} registros com PDV inválido")
        
        pdv_prod_list = aggregated_data_clean[['pdv', 'internal_product_id']].drop_duplicates().values
        print(f"   → {len(pdv_prod_list):,} combinações totais encontradas")
        
        # Otimizar mapeamento de dados históricos
        print("   → Preparando dados históricos de forma otimizada...")
        hist_data_map = {}
        
        # Agrupar dados uma única vez e converter para numpy arrays
        grouped_data = aggregated_data_clean.groupby(['pdv', 'internal_product_id'])
        for (pdv, sku), group in grouped_data:
            # Ordenar e converter para formato otimizado
            sorted_group = group.sort_values('week_of_year')
            
            # Extrair apenas as colunas necessárias em formato numpy para velocidade
            hist_data_optimized = {
                'qty': sorted_group['qty'].values,
                'week_of_year': sorted_group['week_of_year'].values,
                'categorical_data': sorted_group.iloc[-1]  # Apenas último registro para features categóricas
            }
            hist_data_map[(pdv, sku)] = (sorted_group, hist_data_optimized)
        
        valid_tasks = []
        for pdv, sku in pdv_prod_list:
            hist_tuple = hist_data_map.get((pdv, sku))
            if hist_tuple is not None:
                hist, hist_opt = hist_tuple
                # Usar critério otimizado para histórico
                if len(hist_opt['qty']) >= 3:
                    valid_tasks.append((pdv, sku, hist))  # Manter formato original para compatibilidade
        
        print(f"   → {len(valid_tasks):,} combinações válidas (com histórico >= 3 semanas) para predição")
        
        if len(valid_tasks) == 0:
            print("   → ⚠️ Nenhuma combinação atendeu aos critérios de histórico mínimo. Retornando DataFrame vazio.")
            return pd.DataFrame() # Retorna DF vazio para evitar o erro `KeyError`

        # Para datasets muito grandes, usar versão ultra-otimizada
        if len(valid_tasks) > 100000:
            print("   → Dataset muito grande detectado. Usando versão ultra-otimizada...")
            return self._predict_ultra_fast(valid_tasks, weeks)

        all_predictions = []
        
        # Otimizar configuração de paralelização baseada no número de tarefas
        if parallel and len(valid_tasks) > 50:  # Reduzir threshold
            # Usar menos processos para reduzir overhead, mas chunks maiores
            num_processes = min(4, mp.cpu_count() // 2)  # Máximo 4 processos
            chunksize = max(100, len(valid_tasks) // (num_processes * 2))  # Chunks maiores
            print(f"   → Utilizando {num_processes} processos com chunksize {chunksize}")
            
            chunks = [valid_tasks[i:i + chunksize] for i in range(0, len(valid_tasks), chunksize)]
            chunk_args = [(chunk, self.model, self.feature_columns, weeks) for chunk in chunks]
            
            with mp.Pool(processes=num_processes) as pool:
                results = list(tqdm(
                    pool.imap(_process_prediction_chunk, chunk_args), 
                    total=len(chunks), desc="Predições (chunks)"
                ))
            
            for result_list in results:
                all_predictions.extend(result_list)
        else:
            print("   → Usando processamento sequencial otimizado")
            # Usar batch otimizado mesmo no sequencial
            batch_size = 1000  # Processar em lotes
            for i in tqdm(range(0, len(valid_tasks), batch_size), desc="Predições"):
                batch = valid_tasks[i:i + batch_size]
                try:
                    predictions = _predict_batch_optimized(batch, self.model, self.feature_columns, weeks)
                    all_predictions.extend(predictions)
                except Exception:
                    continue
        
        print(f"   → {len(all_predictions):,} predições geradas")
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
                batch_predictions = _predict_batch_optimized(mega_batch, self.model, self.feature_columns, weeks)
                all_predictions.extend(batch_predictions)
            except Exception as e:
                print(f"   → Erro no mega-batch {i//mega_batch_size + 1}: {e}")
                continue
        
        print(f"   → {len(all_predictions):,} predições geradas (ultra-rápido)")
        return pd.DataFrame(all_predictions)

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
    
    # Reordenar colunas para o formato solicitado: semana, pdv, produto, quantidade
    predictions_formatted = pd.DataFrame({
        'semana': predictions_df['week'].astype(int),
        'pdv': predictions_df['pdv'].astype(int),
        'produto': predictions_df['internal_product_id'].astype(int),
        'quantidade': predictions_df['predicted_qty'].round().astype(int)
    })
    
    # Salvar predições se caminho fornecido
    if save_path:
        predictions_formatted.to_csv(save_path, sep=';', index=False, encoding='utf-8')
        print(f"💾 Predições salvas em: {save_path}")
    
    return predictions_formatted