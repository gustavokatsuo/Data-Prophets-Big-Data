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
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

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
                # ATUALIZADO: remover referências a target_enc e embeddings complexos  
                simple_emb_cols = [col for col in hist_data.columns if col.endswith(('_log_freq', '_freq_rank'))]
                
                categorical_features = {}
                if dummy_cols:
                    categorical_features.update(last_row[dummy_cols].to_dict())
                if simple_emb_cols:
                    categorical_features.update(last_row[simple_emb_cols].to_dict())
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

def _calculate_all_features_for_prediction(qty_values, week, categorical_features, feature_columns):
    """
    VERSÃO VETORIZADA - Calcula TODAS as features de forma super otimizada
    Esta função garante que as mesmas features usadas no treino sejam usadas na predição
    """
    features_dict = {'week_of_year': week}
    n_qty = len(qty_values)
    
    if n_qty == 0:
        # Se não há dados, retornar features zeradas + categóricas
        for col in feature_columns:
            if col not in categorical_features and col != 'week_of_year':
                features_dict[col] = 0
        features_dict.update(categorical_features)
        return features_dict
    
    # Converter para numpy array uma única vez para performance
    qty_array = np.asarray(qty_values, dtype=np.float32)
    
    # 🚀 PRÉ-COMPUTAR TUDO DE UMA VEZ
    # Criar dicionário para armazenar rolling features calculadas apenas uma vez
    rolling_cache = {}
    
    # 1️⃣ LAGS VETORIZADOS - calcular todos os lags de uma vez
    lag_features = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8', 'lag_12', 'lag_16', 'lag_24']
    lag_indices = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24]
    
    # Calcular apenas os lags que existem no modelo
    needed_lags = [i for i, lag_name in enumerate(lag_features) if lag_name in feature_columns]
    if needed_lags:
        lag_values = np.zeros(len(lag_indices), dtype=np.float32)
        valid_indices = [i for i in needed_lags if lag_indices[i] <= n_qty]
        if valid_indices:
            lag_positions = [-lag_indices[i] for i in valid_indices]
            lag_values[valid_indices] = qty_array[lag_positions]
        
        for i, lag_name in enumerate(lag_features):
            if lag_name in feature_columns:
                features_dict[lag_name] = float(lag_values[i])
    
    # 2️⃣ ROLLING FEATURES VETORIZADAS - calcular todas as janelas de uma vez
    windows = [4, 8, 12, 16, 20]
    rolling_types = ['rmean', 'rstd', 'rmax', 'rmin', 'rrange', 'rskew', 'rcv']
    
    # Identificar quais windows são necessárias
    needed_windows = []
    needed_types = set()
    for window in windows:
        window_needed = False
        for rtype in rolling_types:
            if f'{rtype}_{window}' in feature_columns:
                needed_types.add(rtype)
                window_needed = True
        if window_needed:
            needed_windows.append(window)
    
    # Calcular rolling features apenas para windows necessárias
    for window in needed_windows:
        if n_qty >= window:
            window_data = qty_array[-window:]
        else:
            window_data = qty_array
        
        # Calcular todas as estatísticas de uma vez
        if len(window_data) > 0:
            if 'rmean' in needed_types:
                rolling_cache[f'rmean_{window}'] = float(np.mean(window_data))
            if 'rstd' in needed_types:
                rolling_cache[f'rstd_{window}'] = float(np.std(window_data)) if len(window_data) > 1 else 0.0
            if 'rmax' in needed_types:
                rolling_cache[f'rmax_{window}'] = float(np.max(window_data))
            if 'rmin' in needed_types:
                rolling_cache[f'rmin_{window}'] = float(np.min(window_data))
            if 'rrange' in needed_types:
                rolling_cache[f'rrange_{window}'] = float(np.max(window_data) - np.min(window_data)) if len(window_data) > 1 else 0.0
            if 'rskew' in needed_types and SCIPY_AVAILABLE and len(window_data) >= 3:
                try:
                    rolling_cache[f'rskew_{window}'] = float(stats.skew(window_data))
                except:
                    rolling_cache[f'rskew_{window}'] = 0.0
            elif 'rskew' in needed_types:
                rolling_cache[f'rskew_{window}'] = 0.0
        else:
            # Valores padrão quando não há dados
            for rtype in needed_types:
                rolling_cache[f'{rtype}_{window}'] = 0.0
    
    # Adicionar features rolling calculadas ao resultado
    for window in windows:
        for rtype in rolling_types:
            col_name = f'{rtype}_{window}'
            if col_name in feature_columns:
                features_dict[col_name] = rolling_cache.get(col_name, 0.0)
    
    # Calcular rcv baseado em rmean e rstd já calculados
    for window in windows:
        rcv_col = f'rcv_{window}'
        if rcv_col in feature_columns:
            mean_val = rolling_cache.get(f'rmean_{window}', 0.0)
            std_val = rolling_cache.get(f'rstd_{window}', 0.0)
            features_dict[rcv_col] = std_val / (mean_val + 0.01)
    
    # 3️⃣ VARIAÇÕES PERCENTUAIS VETORIZADAS
    pct_lags = [1, 2, 4, 8]
    if n_qty >= 1:
        current_val = qty_array[-1]
        for lag in pct_lags:
            col_name = f'pct_change_{lag}'
            if col_name in feature_columns:
                if lag <= n_qty:
                    past_val = qty_array[-lag]
                    denominator = max(abs(past_val), 0.01)
                    features_dict[col_name] = float((current_val - past_val) / denominator)
                else:
                    features_dict[col_name] = 0.0
    else:
        for lag in pct_lags:
            col_name = f'pct_change_{lag}'
            if col_name in feature_columns:
                features_dict[col_name] = 0.0
    
    # 4️⃣ FEATURES DE SAZONALIDADE OTIMIZADAS
    if isinstance(week, pd.Timestamp):
        month = week.month
        quarter = week.quarter
        week_of_month = min(((week - week.replace(day=1)).days // 7) + 1, 5)
        is_end_month = int((week.replace(day=1) + pd.DateOffset(months=1) - week).days <= 7)
        is_start_month = int(week.day <= 7)
        is_end_quarter = int(week.month in [3, 6, 9, 12])
        is_start_quarter = int(week.month in [1, 4, 7, 10])
        is_holiday = int(week.month in [12, 1])
    else:
        # Aproximações numéricas mais eficientes
        month_approx = ((week - 1) // 4.33) % 12 + 1
        month = int(np.clip(month_approx, 1, 12))
        quarter = int(((month - 1) // 3) + 1)
        week_of_month = int(((week - 1) % 4.33) + 1)
        week_in_month = (week - 1) % 4.33
        week_in_quarter = (week - 1) % 13
        is_end_month = int(week_in_month >= 3.5)
        is_start_month = int(week_in_month <= 0.5)
        is_end_quarter = int(week_in_quarter >= 11)
        is_start_quarter = int(week_in_quarter <= 1)
        is_holiday = int(week >= 48)
    
    # Mapear features de sazonalidade
    seasonality_map = {
        'month': month,
        'quarter': quarter,
        'week_of_month': week_of_month,
        'month_sin': np.sin(2 * np.pi * month / 12),
        'month_cos': np.cos(2 * np.pi * month / 12),
        'quarter_sin': np.sin(2 * np.pi * quarter / 4),
        'quarter_cos': np.cos(2 * np.pi * quarter / 4),
        'is_end_of_month': is_end_month,
        'is_start_of_month': is_start_month,
        'is_end_of_quarter': is_end_quarter,
        'is_start_of_quarter': is_start_quarter,
        'is_holiday_season': is_holiday
    }
    
    for col, value in seasonality_map.items():
        if col in feature_columns:
            features_dict[col] = float(value)
    
    # 7️⃣ FEATURES DE TENDÊNCIA VETORIZADAS
    trend_windows = [4, 8, 12]
    for window in trend_windows:
        col_name = f'trend_{window}'
        if col_name in feature_columns:
            if n_qty >= window:
                window_data = qty_array[-window:]
                X = np.arange(len(window_data), dtype=np.float32)
                var_x = np.var(X)
                if var_x > 0:
                    slope = np.cov(X, window_data)[0, 1] / var_x
                    features_dict[col_name] = float(slope)
                else:
                    features_dict[col_name] = 0.0
            else:
                features_dict[col_name] = 0.0
    
    # 8️⃣ FEATURES DE ACELERAÇÃO (baseadas em tendências já calculadas)
    accel_map = {
        'accel_4_8': ('trend_4', 'trend_8'),
        'accel_8_12': ('trend_8', 'trend_12')
    }
    for accel_col, (trend1, trend2) in accel_map.items():
        if accel_col in feature_columns:
            val1 = features_dict.get(trend1, 0.0)
            val2 = features_dict.get(trend2, 0.0)
            features_dict[accel_col] = val1 - val2
    
    # 9️⃣ FEATURES DE MOMENTUM (baseadas em médias já calculadas)
    momentum_map = {
        'momentum_4_8': ('rmean_4', 'rmean_8'),
        'momentum_8_12': ('rmean_8', 'rmean_12'),
        'momentum_4_12': ('rmean_4', 'rmean_12')
    }
    for momentum_col, (short_mean, long_mean) in momentum_map.items():
        if momentum_col in feature_columns:
            short_val = features_dict.get(short_mean, 0.0)
            long_val = features_dict.get(long_mean, 0.0)
            features_dict[momentum_col] = short_val / (long_val + 0.01)
    
    # 🔟 NONZERO FRACTION
    if 'nonzero_frac_8' in feature_columns:
        if n_qty >= 8:
            nonzero_data = qty_array[-8:]
        else:
            nonzero_data = qty_array
        features_dict['nonzero_frac_8'] = float(np.mean(nonzero_data > 0)) if len(nonzero_data) > 0 else 0.0
    
    # 🔗 FEATURES CATEGÓRICAS (adicionar por último)
    features_dict.update(categorical_features)
    
    return features_dict

def _predict_single_fast(pdv, sku, hist_data, model, feature_columns, weeks, categorical_features):
    """Versão ultra-rápida de predição com features categóricas em cache"""
    try:
        predictions = []
        qty_values = hist_data['qty'].values
        
        for week in weeks:
            # USAR A NOVA FUNÇÃO QUE CALCULA TODAS AS FEATURES
            features_dict = _calculate_all_features_for_prediction(
                qty_values, week, categorical_features, feature_columns
            )
            
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

class LightGBMModel:
    """Classe para treinamento e predição com LightGBM"""
    
    def __init__(self, model_params=None, training_params=None):
        self.model_params = model_params or MODEL_PARAMS
        self.training_params = training_params or TRAINING_PARAMS
        self.model = None
        self.feature_names = None
        self.best_iteration = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Treina o modelo LightGBM na escala original"""
        print("🤖 Treinando modelo LightGBM...")
        
        print(f"   → Target range - Original: [{y_train.min():.2f}, {y_train.max():.2f}]")
        
        # Datasets do LightGBM (sem transformação logarítmica)
        dtrain = lgb.Dataset(X_train, label=y_train)
        valid_sets = [dtrain]
        valid_names = ['training']
        
        if X_val is not None and y_val is not None:
            dval = lgb.Dataset(X_val, label=y_val)
            valid_sets.append(dval)
            valid_names.append('valid_1')
            print(f"   → Target validação range: [{y_val.min():.2f}, {y_val.max():.2f}]")

        # Usar configuração padrão do LightGBM sem função objetivo personalizada
        # Remover função objetivo personalizada para evitar problemas de serialização
        if 'objective' in self.model_params and callable(self.model_params['objective']):
            self.model_params = self.model_params.copy()
            self.model_params['objective'] = 'regression'  # Usar regressão padrão
            
        # Usar métrica WMAPE customizada
        def feval_wmape(y_pred, dtrain):
            y_true = dtrain.get_label()
            # Calcular WMAPE na escala original
            wmape = np.sum(np.abs(y_pred - y_true)) / np.sum(np.abs(y_true))
            # Retornar (nome_da_métrica, valor, is_higher_better)
            return ("wmape", wmape, False)

        # Treinar modelo
        self.model = lgb.train(
            self.model_params,
            dtrain,
            valid_sets=valid_sets,
            valid_names=valid_names,
            feval=feval_wmape,
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
            # Fazer predições na escala original
            pred_val = self.predict(X_val)
            
            # Calcular WMAPE na escala original
            wmape = np.sum(np.abs(pred_val - y_val)) / np.sum(np.abs(y_val))
            print(f'📊 WMAPE validação: {wmape:.4f}')
            
            # Calcular MAE para comparação
            mae = np.mean(np.abs(pred_val - y_val))
            print(f'📊 MAE validação: {mae:.4f}')
            
            return wmape
        
        return None
    
    def predict(self, X):
        """Faz predições com o modelo treinado na escala original"""
        if self.model is None:
            raise ValueError("Modelo não foi treinado. Execute train() primeiro.")
        
        # Fazer predição diretamente na escala original
        if self.best_iteration is None:
            pred = self.model.predict(X)
        else:
            # Use num_iteration instead of iteration (deprecated in newer LightGBM versions)
            try:
                pred = self.model.predict(X, num_iteration=self.best_iteration)
            except (TypeError, ValueError):
                try:
                    pred = self.model.predict(X, iteration=self.best_iteration)
                except:
                    pred = self.model.predict(X)
        
        # Garantir que não há valores negativos
        pred = np.maximum(pred, 0)
        
        return pred
    
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
        # ATUALIZADO: usar apenas features simples sem vazamento
        simple_emb_cols = [col for col in hist_data.columns if col.endswith(('_log_freq', '_freq_rank'))]
        
        categorical_features = {}
        if dummy_cols:
            categorical_features.update(last_row[dummy_cols].to_dict())
        if simple_emb_cols:
            categorical_features.update(last_row[simple_emb_cols].to_dict())
        
        # Pre-computar lag indices para otimização
        lag_indices = np.array([1, 2, 3, 4, 8, 12])
        
        for week in weeks:
            # USAR A NOVA FUNÇÃO QUE CALCULA TODAS AS FEATURES
            features_dict = _calculate_all_features_for_prediction(
                lags, week, categorical_features, self.feature_columns
            )
            
            # Criar DataFrame e selecionar apenas as features que o modelo conhece
            X_row = pd.DataFrame([features_dict])
            
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
        
        # Selecionar apenas combinações com histórico de vendas real (qty > 0) - Tentativa de reduzir o número de predições desnecessárias
        sales_by_combination = aggregated_data_clean.groupby(['pdv', 'internal_product_id'])['qty'].sum()
        active_combinations = sales_by_combination[sales_by_combination > 0].reset_index()

        pdv_prod_list = active_combinations[['pdv', 'internal_product_id']].values
        print(f"   → {len(pdv_prod_list):,} combinações com vendas históricas selecionadas para previsão")
        
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
                if len(hist_opt['qty']) >= 8:
                    valid_tasks.append((pdv, sku, hist))  # Manter formato original para compatibilidade

        print(f"   → {len(valid_tasks):,} combinações válidas (com histórico >= 8 semanas) para predição")

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
    
    predictions_df['predicted_qty'] = predictions_df['predicted_qty'].apply(lambda x: x if x >= 0.5 else 0) # Thresholding

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