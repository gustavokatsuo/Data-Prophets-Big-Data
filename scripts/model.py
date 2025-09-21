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

def _process_prediction_chunk(chunk_args):
    """Função auxiliar para processar chunks de predições em paralelo"""
    chunk_data, model, feature_columns, weeks = chunk_args
    chunk_results = []
    for pdv, sku, hist_data in chunk_data:
        # A função de predição agora pode imprimir erros detalhados
        predictions = _predict_single_optimized_static(pdv, sku, hist_data, model, feature_columns, weeks)
        chunk_results.extend(predictions)
    return chunk_results

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
                        col in ['categoria', 'premise', 'subcategoria', 'tipos', 'label', 'fabricante', 'gross']):
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
        
        # Lags
        for lag in [1, 2, 3, 4, 8, 12]:
            features.append(qty_values[-lag] if len(qty_values) >= lag else 0)

        # Dados para rolling (todos exceto o mais recente)
        rolling_data = qty_values[:-1] if len(qty_values) > 1 else qty_values

        # Rolling means/std sobre os últimos 4 pontos *do passado*
        if len(rolling_data) >= 4:
            window = rolling_data[-4:]
            features.append(np.mean(window))  # rmean_4
            features.append(np.std(window))   # rstd_4
        else: # Se não houver dados suficientes, usar a média/std de tudo que tiver
            features.append(np.mean(rolling_data) if len(rolling_data) > 0 else 0)
            features.append(np.std(rolling_data) if len(rolling_data) > 1 else 0)
        
        # Fração de não zeros sobre os últimos 8 pontos *do passado*
        if len(rolling_data) >= 8:
            features.append(np.mean(rolling_data[-8:] > 0))  # nonzero_frac_8
        else:
            features.append(np.mean(rolling_data > 0) if len(rolling_data) > 0 else 0)
        
        features.append(qty_values[-1] * 10 if len(qty_values) > 0 else 0)
        
        # Adicionar features categóricas do último registro
        # Estas features são constantes para cada combinação PDV-produto
        last_row = data.iloc[-1]
        
        # Features dummy - verificar se existem nas colunas
        for col in data.columns:
            if col.startswith(('categoria_', 'premise_')):
                features.append(last_row[col] if col in data.columns else 0)
        
        # Features de embedding - verificar se existem nas colunas  
        for col in data.columns:
            if col.endswith(('_target_enc', '_emb_0', '_emb_1', '_emb_2', '_emb_3', '_emb_4', 
                           '_emb_5', '_emb_6', '_emb_7', '_emb_8', '_emb_9')):
                features.append(last_row[col] if col in data.columns else 0)
        
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
        
        lags = hist_data['qty'].values.tolist()
        last_gross = hist_data['gross'].values.tolist()
        
        # Extrair features categóricas do último registro (são constantes para cada PDV-produto)
        last_row = hist_data.iloc[-1]
        categorical_features = {}
        
        # Features dummy
        for col in hist_data.columns:
            if col.startswith(('categoria_', 'premise_')):
                categorical_features[col] = last_row[col]
        
        # Features de embedding
        for col in hist_data.columns:
            if col.endswith(('_target_enc', '_emb_0', '_emb_1', '_emb_2', '_emb_3', '_emb_4', 
                           '_emb_5', '_emb_6', '_emb_7', '_emb_8', '_emb_9')):
                categorical_features[col] = last_row[col]
        
        for week in weeks:
            feat = {'week_of_year': week, 'gross': last_gross[-1] if last_gross else 0}
            
            # Features de lag
            for lag in [1, 2, 3, 4, 8, 12]:
                feat[f'lag_{lag}'] = lags[-lag] if len(lags) >= lag else 0
            
            # Features rolling
            recent_values = lags[-4:] if len(lags) >= 4 else lags
            feat['rmean_4'] = np.mean(recent_values) if recent_values else 0
            feat['rstd_4'] = np.std(recent_values) if len(recent_values) > 1 else 0
            
            nonzero_values = lags[-8:] if len(lags) >= 8 else lags
            feat['nonzero_frac_8'] = np.mean([1 if x > 0 else 0 for x in nonzero_values]) if nonzero_values else 0
            
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
            
            lags.append(pred)
            if last_gross:
                last_gross.append(last_gross[-1])
        
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
        
        hist_data_map = {
            (pdv, sku): group.sort_values('week_of_year')
            for (pdv, sku), group in aggregated_data_clean.groupby(['pdv', 'internal_product_id'])
        }
        
        valid_tasks = []
        for pdv, sku in pdv_prod_list:
            hist = hist_data_map.get((pdv, sku))
            # Adicionado comentário explicativo sobre o filtro que pode causar predições vazias.
            if hist is not None and not hist.empty and len(hist) >= 3:
                valid_tasks.append((pdv, sku, hist))
        
        print(f"   → {len(valid_tasks):,} combinações válidas (com histórico >= 3 semanas) para predição")
        
        if len(valid_tasks) == 0:
            print("   → ⚠️ Nenhuma combinação atendeu aos critérios de histórico mínimo. Retornando DataFrame vazio.")
            return pd.DataFrame() # Retorna DF vazio para evitar o erro `KeyError`

        all_predictions = []
        
        if parallel and len(valid_tasks) > 100:
            num_processes = min(mp.cpu_count(), len(valid_tasks))
            chunksize = max(1, len(valid_tasks) // (num_processes * 4))
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
            for pdv, sku, hist in tqdm(valid_tasks, desc="Predições"):
                try:
                    predictions = _predict_single_optimized_static(pdv, sku, hist, self.model, self.feature_columns, weeks)
                    all_predictions.extend(predictions)
                except Exception:
                    continue
        
        print(f"   → {len(all_predictions):,} predições geradas")
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
        'quantidade': predictions_df['predicted_qty'].round().astype(int)  # ARREDONDAR PARA INTEIRO
    })
    
    # Salvar predições se caminho fornecido
    if save_path:
        predictions_formatted.to_csv(save_path, sep=';', index=False, encoding='utf-8')
        print(f"💾 Predições salvas em: {save_path}")
    
    return predictions_formatted