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

def _calculate_features_vectorized_batch(hist_data_dict, feature_columns, weeks):
    """
    Calcula features para múltiplas combinações de forma vetorizada com otimizações avançadas
    """
    print("   → Calculando features de forma vetorizada ultra-otimizada...")
    
    # Pre-alocar arrays para melhor performance
    n_combinations = len(hist_data_dict)
    n_weeks = len(weeks)
    total_samples = n_combinations * n_weeks
    
    # Estimar tamanho de features baseado no primeiro item
    first_key = next(iter(hist_data_dict))
    first_hist = hist_data_dict[first_key]
    
    # Calcular número de features categóricas
    cat_features_count = 0
    for col in first_hist.columns:
        if (col.startswith(('categoria_', 'premise_')) or 
            col.endswith(('_target_enc', '_emb_0', '_emb_1', '_emb_2', '_emb_3', '_emb_4', 
                         '_emb_5', '_emb_6', '_emb_7', '_emb_8', '_emb_9'))):
            cat_features_count += 1
    
    # Número total de features: week + 6 lags + 2 rolling + 1 nonzero_frac + categóricas
    n_features = 1 + 6 + 2 + 1 + cat_features_count
    
    # Pre-alocar arrays numpy (muito mais rápido)
    feature_matrix = np.zeros((total_samples, n_features), dtype=np.float32)
    metadata = np.zeros((total_samples, 4), dtype=np.int32)  # pdv, sku, week, week_idx
    
    # Cache para features categóricas (computar uma vez por combinação)
    categorical_cache = {}
    categorical_columns = []
    
    # Identificar colunas categóricas de forma otimizada
    for col in first_hist.columns:
        if (col.startswith(('categoria_', 'premise_')) or 
            col.endswith(('_target_enc', '_emb_0', '_emb_1', '_emb_2', '_emb_3', '_emb_4', 
                         '_emb_5', '_emb_6', '_emb_7', '_emb_8', '_emb_9'))):
            categorical_columns.append(col)
    
    categorical_columns.sort()  # Para ordem consistente
    
    sample_idx = 0
    
    # Processar cada combinação de forma otimizada
    for (pdv, sku), hist_data in hist_data_dict.items():
        try:
            if len(hist_data) < 1:
                continue
            
            # Cache de features categóricas (uma vez por combinação)
            if (pdv, sku) not in categorical_cache:
                last_row = hist_data.iloc[-1]
                cat_values = np.array([
                    last_row[col] if col in hist_data.columns else 0.0 
                    for col in categorical_columns
                ], dtype=np.float32)
                categorical_cache[(pdv, sku)] = cat_values
            
            cat_features = categorical_cache[(pdv, sku)]
            
            # Preparar dados de quantidade como numpy array
            qty_values = hist_data['qty'].values.astype(np.float32)
            
            # Vectorizar cálculo para todas as semanas desta combinação
            weeks_array = np.array(weeks, dtype=np.int32)
            
            for week_idx, week in enumerate(weeks_array):
                if sample_idx >= total_samples:
                    break
                
                # Features temporais básicas
                feature_matrix[sample_idx, 0] = week  # week_of_year
                
                # Lags vetorizados (mais rápido que loop)
                lag_indices = [1, 2, 3, 4, 8, 12]
                for i, lag in enumerate(lag_indices):
                    if len(qty_values) >= lag:
                        feature_matrix[sample_idx, 1 + i] = qty_values[-lag]
                    # else: já é zero por padrão
                
                # Rolling features otimizados
                rolling_data = qty_values[:-1] if len(qty_values) > 1 else qty_values
                
                if len(rolling_data) >= 4:
                    window = rolling_data[-4:]
                    feature_matrix[sample_idx, 7] = np.mean(window)  # rmean_4
                    feature_matrix[sample_idx, 8] = np.std(window)   # rstd_4
                elif len(rolling_data) > 0:
                    feature_matrix[sample_idx, 7] = np.mean(rolling_data)
                    if len(rolling_data) > 1:
                        feature_matrix[sample_idx, 8] = np.std(rolling_data)
                
                # Nonzero fraction otimizado
                if len(rolling_data) >= 8:
                    feature_matrix[sample_idx, 9] = np.mean(rolling_data[-8:] > 0)
                elif len(rolling_data) > 0:
                    feature_matrix[sample_idx, 9] = np.mean(rolling_data > 0)
                
                # Features categóricas (do cache)
                if len(cat_features) > 0:
                    feature_matrix[sample_idx, 10:10+len(cat_features)] = cat_features
                
                # Metadata
                metadata[sample_idx] = [pdv, sku, week, week_idx]
                sample_idx += 1
                
        except Exception as e:
            print(f"Erro ao processar PDV {pdv}, SKU {sku}: {e}")
            continue
    
    # Truncar arrays para o tamanho real usado
    feature_matrix = feature_matrix[:sample_idx]
    metadata_list = [(int(row[0]), int(row[1]), int(row[2]), int(row[3])) 
                     for row in metadata[:sample_idx]]
    
    print(f"   → {sample_idx:,} samples de features calculados")
    return feature_matrix, metadata_list

def _predict_batch_vectorized(model, feature_matrix, metadata, feature_columns):
    """
    Faz predições em lote de forma ultra-otimizada
    """
    print(f"   → Fazendo predições em lote para {len(feature_matrix):,} amostras...")
    
    if len(feature_matrix) == 0:
        return []
    
    # Garantir que o número de features está correto
    if feature_matrix.shape[1] != len(feature_columns):
        print(f"⚠️ Mismatch de features: matrix tem {feature_matrix.shape[1]}, esperado {len(feature_columns)}")
        return []
    
    # Predição em lote ultra-otimizada
    try:
        # Para modelos LightGBM, usar predict diretamente com numpy é mais rápido
        predictions = model.predict(feature_matrix)
        
        # Operação vetorizada para garantir valores não-negativos
        predictions = np.maximum(predictions, 0.0)
        
        # Construção otimizada de resultados usando list comprehension
        results = [
            {
                'pdv': int(pdv),
                'internal_product_id': int(sku),
                'week': int(week),
                'predicted_qty': float(pred)
            }
            for pred, (pdv, sku, week, week_idx) in zip(predictions, metadata)
        ]
        
        print(f"   → {len(results):,} predições processadas com sucesso")
        return results
        
    except Exception as e:
        print(f"⚠️ Erro durante predição em lote: {e}")
        return []

def _process_prediction_chunk_vectorized(chunk_args):
    """Função auxiliar para processar chunks de predições de forma vetorizada"""
    chunk_data, model, feature_columns, weeks = chunk_args
    
    # Converter chunk para dict
    hist_data_dict = {(pdv, sku): hist_data for pdv, sku, hist_data in chunk_data}
    
    # Calcular features vetorizadas
    feature_matrix, metadata = _calculate_features_vectorized_batch(hist_data_dict, feature_columns, weeks)
    
    # Predições vetorizadas
    if len(feature_matrix) > 0:
        return _predict_batch_vectorized(model, feature_matrix, metadata, feature_columns)
    else:
        return []

class FastVectorizedPredictor:
    """Preditor ultra-otimizado com operações vetorizadas e gestão de memória"""
    
    def __init__(self, model, feature_columns):
        self.model = model
        self.feature_columns = feature_columns
        
    def predict_all_combinations_vectorized(self, aggregated_data, weeks=None, parallel=True, batch_size=2000):
        """
        Prediz para todas as combinações usando operações completamente vetorizadas com otimizações avançadas
        """
        print("🚀 Gerando predições com vetorização ultra-otimizada...")
        weeks = weeks or PREDICTION_WEEKS
        
        # Filtrar dados inválidos de forma otimizada
        print("   → Filtrando e preparando dados...")
        original_count = len(aggregated_data)
        
        # Filtro vetorizado mais eficiente
        is_numeric_pdv = pd.to_numeric(aggregated_data['pdv'], errors='coerce').notna()
        aggregated_data_clean = aggregated_data[is_numeric_pdv].copy()
        aggregated_data_clean['pdv'] = aggregated_data_clean['pdv'].astype(np.int32)
        aggregated_data_clean['internal_product_id'] = aggregated_data_clean['internal_product_id'].astype(np.int32)
        
        filtered_count = len(aggregated_data_clean)
        if original_count != filtered_count:
            print(f"   → Removidos {original_count - filtered_count:,} registros com PDV inválido")
        
        # Preparar dados históricos de forma mais eficiente
        print("   → Agrupando dados históricos...")
        hist_data_map = {}
        
        # Usar groupby mais eficiente
        grouped = aggregated_data_clean.groupby(['pdv', 'internal_product_id'], sort=False)
        
        for (pdv, sku), group in grouped:
            hist = group.sort_values('week_of_year', kind='mergesort')  # mergesort é estável e rápido
            if len(hist) >= 3:  # Filtro de histórico mínimo
                hist_data_map[(pdv, sku)] = hist
        
        print(f"   → {len(hist_data_map):,} combinações válidas para predição")
        
        if len(hist_data_map) == 0:
            print("   → ⚠️ Nenhuma combinação válida encontrada.")
            return pd.DataFrame()
        
        # Decisão inteligente sobre processamento
        total_samples = len(hist_data_map) * len(weeks)
        print(f"   → Total de samples estimados: {total_samples:,}")
        
        all_predictions = []
        
        if parallel and len(hist_data_map) > batch_size and total_samples > 10000:
            print(f"   → Processamento paralelo com batch_size={batch_size}")
            all_predictions = self._process_parallel(hist_data_map, weeks, batch_size)
        else:
            print("   → Processamento sequencial ultra-otimizado")
            all_predictions = self._process_sequential(hist_data_map, weeks)
        
        if not all_predictions:
            print("   → ⚠️ Nenhuma predição foi gerada.")
            return pd.DataFrame()
        
        # Criar DataFrame otimizado
        print("   → Criando DataFrame de resultados...")
        result_df = pd.DataFrame(all_predictions)
        
        print(f"✅ {len(all_predictions):,} predições geradas com sucesso")
        return result_df
    
    def _process_sequential(self, hist_data_map, weeks):
        """Processamento sequencial ultra-otimizado"""
        feature_matrix, metadata = _calculate_features_vectorized_batch(
            hist_data_map, self.feature_columns, weeks
        )
        
        if len(feature_matrix) > 0:
            return _predict_batch_vectorized(
                self.model, feature_matrix, metadata, self.feature_columns
            )
        return []
    
    def _process_parallel(self, hist_data_map, weeks, batch_size):
        """Processamento paralelo otimizado"""
        # Dividir em chunks otimizados
        hist_items = list(hist_data_map.items())
        chunks = []
        
        for i in range(0, len(hist_items), batch_size):
            chunk_items = hist_items[i:i + batch_size]
            chunk_data = [(pdv, sku, hist_data) for (pdv, sku), hist_data in chunk_items]
            chunks.append(chunk_data)
        
        chunk_args = [(chunk, self.model, self.feature_columns, weeks) for chunk in chunks]
        
        # Usar número otimizado de processos
        num_processes = min(mp.cpu_count() - 1, len(chunks), 8)  # Máximo 8 processos
        print(f"   → Usando {num_processes} processos para {len(chunks)} chunks")
        
        all_predictions = []
        
        try:
            with mp.Pool(processes=num_processes) as pool:
                results = list(tqdm(
                    pool.imap(_process_prediction_chunk_vectorized, chunk_args), 
                    total=len(chunks), 
                    desc="Predições vetorizadas",
                    disable=False
                ))
            
            for result_list in results:
                if result_list:  # Verificar se não está vazio
                    all_predictions.extend(result_list)
                    
        except Exception as e:
            print(f"⚠️ Erro no processamento paralelo: {e}")
            print("   → Tentando processamento sequencial...")
            all_predictions = self._process_sequential(hist_data_map, weeks)
        
        return all_predictions
    
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
        # Criar instância do preditor vetorizado para melhor performance
        self.vectorized_predictor = FastVectorizedPredictor(model, feature_columns)
    
    def predict_single_combination(self, pdv, sku, hist_data, weeks=None):
        """Prediz vendas para uma combinação PDV-produto específica (mantida para compatibilidade)"""
        weeks = weeks or PREDICTION_WEEKS
        predictions = []
        
        # VALIDAR PDV ANTES DO PROCESSAMENTO
        try:
            if isinstance(pdv, str) and not pdv.isdigit():
                print(f"⚠️ PDV inválido encontrado: {pdv}. Pulando...")
                return []
            
            pdv = int(pdv)
            sku = int(sku)
        except (ValueError, TypeError) as e:
            print(f"⚠️ Erro ao converter PDV/SKU: pdv={pdv}, sku={sku}. Erro: {e}")
            return []
        
        # Usar o preditor vetorizado mesmo para uma única combinação
        temp_data = pd.DataFrame()
        temp_data = pd.concat([temp_data, hist_data], ignore_index=True)
        temp_data['pdv'] = pdv
        temp_data['internal_product_id'] = sku
        
        result_df = self.vectorized_predictor.predict_all_combinations_vectorized(
            temp_data, weeks=weeks, parallel=False, batch_size=1
        )
        
        return result_df.to_dict('records') if not result_df.empty else []
    
    def predict_all_combinations(self, aggregated_data, weeks=None, parallel=True):
        """Prediz para todas as combinações PDV-produto usando vetorização otimizada"""
        return self.vectorized_predictor.predict_all_combinations_vectorized(
            aggregated_data, weeks=weeks, parallel=parallel, batch_size=2000
        )

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
    """Função principal para gerar predições usando vetorização otimizada"""
    print("🚀 Iniciando geração de predições com otimizações vetorizadas...")
    
    predictor = Predictor(model, data_dict['feature_columns'])
    
    # Usar o preditor vetorizado diretamente para máxima performance
    predictions_df = predictor.vectorized_predictor.predict_all_combinations_vectorized(
        data_dict['aggregated_data'], 
        weeks=weeks,
        parallel=parallel,
        batch_size=3000  # Batch maior para máxima eficiência
    )
    
    if predictions_df.empty:
        print("⚠️ Nenhuma predição foi gerada.")
        return pd.DataFrame()
    
    # Reordenar colunas para o formato solicitado: semana, pdv, produto, quantidade
    predictions_formatted = pd.DataFrame({
        'semana': predictions_df['week'].astype(int),
        'pdv': predictions_df['pdv'].astype(int),
        'produto': predictions_df['internal_product_id'].astype(int),
        'quantidade': predictions_df['predicted_qty'].round().astype(int)  # ARREDONDAR PARA INTEIRO
    })
    
    # Salvar predições se caminho fornecido
    if save_path:
        predictions_formatted.to_csv(save_path, sep=';', index=False, encoding='utf-8')
        print(f"💾 Predições salvas em: {save_path}")
    
    # Estatísticas de performance
    total_predictions = len(predictions_formatted)
    total_combinations = predictions_formatted[['pdv', 'produto']].drop_duplicates().shape[0]
    weeks_predicted = predictions_formatted['semana'].nunique()
    
    print(f"✅ Predições geradas com sucesso:")
    print(f"   → Total de predições: {total_predictions:,}")
    print(f"   → Combinações PDV-Produto: {total_combinations:,}")
    print(f"   → Semanas preditas: {weeks_predicted}")
    
    return predictions_formatted