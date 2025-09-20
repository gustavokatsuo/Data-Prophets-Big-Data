# Módulo para treinamento e predição com LightGBM
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import pickle
import os
from config import MODEL_PARAMS, TRAINING_PARAMS, PREDICTION_WEEKS
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
        
        return self.model.predict(X, num_iteration=self.best_iteration)
    
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
        """Prediz para todas as combinações PDV-produto"""
        from concurrent.futures import ProcessPoolExecutor
        from tqdm import tqdm
        from multiprocessing import cpu_count
        
        print("🔮 Gerando predições...")
        weeks = weeks or PREDICTION_WEEKS
        
        # Obter combinações únicas
        pdv_prod_list = aggregated_data[['pdv', 'internal_product_id']].drop_duplicates().to_records(index=False)
        
        # Pré-agrupar dados históricos
        hist_data_map = {
            (pdv, sku): group.sort_values('week_of_year')
            for (pdv, sku), group in aggregated_data.groupby(['pdv', 'internal_product_id'])
        }
        
        all_predictions = []
        
        if parallel:
            # Processamento paralelo
            tasks = []
            for pdv, sku in pdv_prod_list:
                hist = hist_data_map.get((pdv, sku), pd.DataFrame())
                if not hist.empty:
                    tasks.append((pdv, sku, hist, self.model, self.feature_columns, weeks))
            
            with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
                results = list(tqdm(
                    executor.map(self._predict_single_task, tasks), 
                    total=len(tasks),
                    desc="Predições"
                ))
            
            # Achatar resultados
            for result_list in results:
                all_predictions.extend(result_list)
        else:
            # Processamento sequencial
            for i, (pdv, sku) in enumerate(tqdm(pdv_prod_list, desc="Predições")):
                hist = hist_data_map.get((pdv, sku), pd.DataFrame())
                if not hist.empty:
                    predictions = self.predict_single_combination(pdv, sku, hist, weeks)
                    all_predictions.extend(predictions)
        
        return pd.DataFrame(all_predictions)
    
    @staticmethod
    def _predict_single_task(args):
        """Task para processamento paralelo"""
        pdv, sku, hist_data, model, feature_columns, weeks = args
        predictor = Predictor(model, feature_columns)
        return predictor.predict_single_combination(pdv, sku, hist_data, weeks)

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
    predictor = Predictor(model.model, data_dict['feature_columns'])
    
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