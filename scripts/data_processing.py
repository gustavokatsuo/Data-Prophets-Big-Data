# Módulo para tratamento e processamento de dados
import pandas as pd
import numpy as np
from tqdm import tqdm
from .config import LAG_FEATURES, OUTLIER_PARAMS, DUMMY_FEATURES, EMBEDDING_FEATURES, EMBEDDING_CONFIG
import warnings
import multiprocessing as mp
from functools import partial
from sklearn.preprocessing import LabelEncoder, StandardScaler
try:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Embedding, Flatten
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.utils import to_categorical
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')  # Reduzir logs do TensorFlow
    # Configurar TensorFlow para usar menos memória
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU memory config error: {e}")
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow não disponível. Usando método PCA como fallback.")
    from sklearn.decomposition import PCA
    TENSORFLOW_AVAILABLE = False
warnings.filterwarnings('ignore')

def _process_outlier_group_wrapper(args):
    """Wrapper function para desempacotar argumentos do multiprocessing."""
    name_group_tuple, outlier_params = args
    name, group = name_group_tuple
    group.name = name
    return _process_outlier_group(group, **outlier_params)

def _process_outlier_group(group, rolling_window, z_threshold, treatment, adaptive_threshold, min_observations, absolute_cap_multiplier=5):
    """Função auxiliar para processar outliers em um único grupo (para paralelização)."""
    try:
        (pdv, product) = group.name
        n_obs = len(group)
        
        # Early exit para grupos muito pequenos
        if n_obs < 3:
            return group, {
                'pdv': pdv, 'product': product, 'n_obs': n_obs, 'mean_qty': 0,
                'std_qty': 0, 'cv': 0, 'zero_pct': 0,
                'threshold_used': z_threshold, 'n_outliers': 0, 'outlier_pct': 0
            }
        
        # Usar valores numpy para melhor performance
        qty_values = group['qty'].values
        
        # Calcular características básicas do grupo usando numpy
        mean_qty = np.mean(qty_values)
        std_qty = np.std(qty_values) or 1.0
        zero_pct = np.mean(qty_values == 0)
        cv = std_qty / (mean_qty + 1e-8)

        threshold = z_threshold
        
        # Lógica de threshold adaptativo (otimizada)
        if adaptive_threshold:
            if n_obs < min_observations:
                threshold *= 1.3 if cv > 2.0 else (1.2 if zero_pct > 0.7 else 1.0)
            else:
                if cv > 2.0: threshold *= 1.5
                elif cv > 1.0: threshold *= 1.2
                else: threshold *= 0.8
                if zero_pct > 0.7: threshold *= 1.3
                if mean_qty > 100: threshold *= 0.9

        # Calcular outliers de forma otimizada
        if n_obs < min_observations:
            # Método simples para grupos pequenos
            outliers_mask = np.abs(qty_values - mean_qty) > (threshold * std_qty)
        else:
            # Método rolling otimizado usando pandas (mais confiável)
            group_sorted = group.sort_values('week_of_year').copy()
            
            # Usar pandas rolling que é mais robusto
            rolling_mean = group_sorted['qty'].shift(1).rolling(rolling_window, min_periods=3).mean().fillna(mean_qty)
            rolling_std = group_sorted['qty'].shift(1).rolling(rolling_window, min_periods=3).std().fillna(std_qty)
            rolling_std = rolling_std.replace(0, std_qty)
            
            # Calcular Z-scores
            z_scores = np.abs((group_sorted['qty'] - rolling_mean) / rolling_std)
            outliers_mask_sorted = z_scores > threshold
            
            # Calcular Z-scores (outliers_mask_sorted é uma pd.Series com o índice ordenado)
            z_scores = np.abs((group_sorted['qty'] - rolling_mean) / rolling_std)
            outliers_mask_sorted = z_scores > threshold

            # Mapear de volta para a ordem original usando reindex (muito mais rápido)
            outliers_mask = outliers_mask_sorted.reindex(group.index).fillna(False).values

        n_outliers = np.sum(outliers_mask)
        
        group_stats = {
            'pdv': pdv, 'product': product, 'n_obs': n_obs, 'mean_qty': mean_qty,
            'std_qty': std_qty, 'cv': cv, 'zero_pct': zero_pct,
            'threshold_used': threshold, 'n_outliers': n_outliers,
            'outlier_pct': (n_outliers / n_obs) * 100 if n_obs > 0 else 0
        }

        # Aplicar tratamento apenas se houver outliers
        if n_outliers > 0:
            group_copy = group.copy()  # Só copiar se necessário
            
            # Aplicar cap absoluto primeiro (mais agressivo)
            absolute_cap = mean_qty * absolute_cap_multiplier
            extreme_mask = qty_values > absolute_cap
            if np.any(extreme_mask):
                group_copy.loc[extreme_mask, 'qty'] = absolute_cap
            
            if treatment == 'winsorize':
                # Usar percentis mais conservadores
                p02, p98 = np.percentile(qty_values, [2, 98])
                # Garantir que os limites são razoáveis
                p02 = max(0, p02)
                p98 = min(p98, mean_qty * 3)  # Cap adicional: máximo 3x a média
                
                group_copy.loc[outliers_mask, 'qty'] = np.clip(group_copy.loc[outliers_mask, 'qty'], p02, p98)
            
            elif treatment == 'cap':
                # Calcular limites de forma mais conservadora
                if n_obs >= min_observations:
                    # Usar percentil 90 como limite superior mais seguro (era 95)
                    upper_limit = min(np.percentile(qty_values, 90), mean_qty + 1.5 * std_qty)
                    lower_limit = max(0, np.percentile(qty_values, 10))
                else:
                    upper_limit = mean_qty + 1.5 * std_qty
                    lower_limit = max(0, mean_qty - 1.5 * std_qty)
                
                # Aplicar cap apenas para reduzir outliers, nunca aumentar
                outlier_values = group_copy.loc[outliers_mask, 'qty']
                capped_values = np.where(outlier_values > upper_limit, upper_limit, 
                                       np.where(outlier_values < lower_limit, lower_limit, outlier_values))
                group_copy.loc[outliers_mask, 'qty'] = capped_values

            elif treatment == 'remove':
                group_copy = group_copy[~outliers_mask]
            
            return group_copy, group_stats
        else:
            return group, group_stats

    except Exception as e:
        # Retornar o grupo original e estatísticas vazias em caso de erro
        return group, {}

class DataProcessor:
    """Classe para processamento e engenharia de features dos dados"""
    
    def __init__(self):
        self.df_raw = None
        self.df_agg = None
        self.feature_columns = None
        self.dummy_encoders = {}
        self.label_encoders = {}
        self.autoencoder_transformers = {}  # Para armazenar autoencoders treinados
        self.categorical_features = []
        
    def load_data(self, file_path):
        """Carrega dados do arquivo parquet"""
        print("📊 Carregando dados...")
        self.df_raw = pd.read_parquet(file_path)
        self.df_raw['quantity'] = self.df_raw['quantity'].fillna(0)
        
        # Remover coluna semana_do_ano se existir e criar week_of_year baseado em datas
        self.df_raw = self.df_raw.drop(columns=['semana_do_ano'], errors='ignore')

        # Remove PDCs inválidos
        self.df_raw = self.df_raw[self.df_raw['pdv'].notnull() & (self.df_raw['pdv'] != '') & (self.df_raw['pdv'] != 'desconhecido')]
        self.df_raw['pdv'] = self.df_raw['pdv'].astype(int)
        min_date = self.df_raw['transaction_date'].min()
        self.df_raw['week_of_year'] = (self.df_raw['transaction_date'] - min_date).dt.days // 7 + 1
        
        # GAMBIARRA ESPECÍFICA: Remoção da semana 37 (dados inflacionados ~60x)
        # Esta semana tem valores anômalos que não representam vendas reais - REMOVER COMPLETAMENTE
        if 37 in self.df_raw['week_of_year'].values:
            print("⚠️ GAMBIARRA DETECTADA: Semana 37 encontrada com dados inflacionados!")
            print("   → Aplicando correção RADICAL: REMOVENDO semana 37 e interpolando valores")
            
            week_37_records = (self.df_raw['week_of_year'] == 37).sum()
            
            # REMOVER COMPLETAMENTE todos os dados da semana 37
            self.df_raw = self.df_raw[self.df_raw['week_of_year'] != 37].copy()
            
            print(f"   → Semana 37 REMOVIDA: {week_37_records:,} registros eliminados")
            print(f"   → Interpolação será feita automaticamente durante a agregação")
        
        
        print(f"   → Dados carregados: {len(self.df_raw):,} registros")
        return self.df_raw
    
    def aggregate_data(self):
        """Agrega dados por semana/PDV/produto"""
        print("🔄 Agregando dados por semana/PDV/produto...")
        
        # Verificar quais colunas categóricas existem nos dados
        available_categorical = [col for col in ['categoria', 'premise', 'subcategoria', 'tipos', 'label', 'fabricante'] 
                               if col in self.df_raw.columns]
        
        print(f"   → Colunas categóricas disponíveis: {available_categorical}")
        
        # Preparar dicionário de agregação
        agg_dict = {
            'quantity': 'sum',
            'gross_value': 'sum'
        }
        
        # Para colunas categóricas, usar 'first'
        for col in available_categorical:
            agg_dict[col] = 'first'
        
        # Agregação principal
        self.df_agg = self.df_raw.groupby(['week_of_year', 'pdv', 'internal_product_id'], as_index=False).agg(agg_dict)
        
        # Renomear colunas principais
        self.df_agg = self.df_agg.rename(columns={'quantity': 'qty', 'gross_value': 'gross'})

        # Garantir que não há quantidades negativas, desconsidera devoluções porém não vicia o modelo
        self.df_agg['qty'] = self.df_agg['qty'].clip(lower=0)
        
        # INTERPOLAÇÃO: Preencher lacuna da semana 37 removida
        if 37 not in self.df_agg['week_of_year'].values and 36 in self.df_agg['week_of_year'].values and 38 in self.df_agg['week_of_year'].values:
            print("🔧 INTERPOLAÇÃO: Preenchendo lacuna da semana 37...")
            self._interpolate_missing_week(37)
        
        print(f"   → Dados agregados iniciais: {len(self.df_agg):,} registros")
        print(f"   → Dados agregados finais: {len(self.df_agg):,} registros")
        print(f"   → PDVs únicos: {self.df_agg['pdv'].nunique():,}")
        print(f"   → Produtos únicos: {self.df_agg['internal_product_id'].nunique():,}")
        print(f"   → Range de semanas: {self.df_agg['week_of_year'].min()}-{self.df_agg['week_of_year'].max()}")
        
        return self.df_agg
    
    def _interpolate_missing_week(self, missing_week):
        """Interpola dados para uma semana ausente usando média das semanas vizinhas"""
        print(f"   → Interpolando dados para semana {missing_week}...")
        
        # Identificar semanas vizinhas
        prev_week = missing_week - 1
        next_week = missing_week + 1
        
        # Buscar dados das semanas vizinhas
        prev_data = self.df_agg[self.df_agg['week_of_year'] == prev_week].copy()
        next_data = self.df_agg[self.df_agg['week_of_year'] == next_week].copy()
        
        if prev_data.empty or next_data.empty:
            print(f"   → Aviso: Não foi possível interpolar semana {missing_week} - semanas vizinhas ausentes")
            return
        
        # Fazer merge das semanas vizinhas para interpolar
        interpolated = prev_data.merge(
            next_data[['pdv', 'internal_product_id', 'qty', 'gross']],
            on=['pdv', 'internal_product_id'],
            how='outer',
            suffixes=('_prev', '_next')
        )
        
        # Calcular valores interpolados (média simples)
        interpolated['qty'] = (
            interpolated['qty_prev'].fillna(0) + interpolated['qty_next'].fillna(0)
        ) / 2
        interpolated['gross'] = (
            interpolated['gross_prev'].fillna(0) + interpolated['gross_next'].fillna(0)
        ) / 2
        
        # Preparar DataFrame interpolado
        interpolated['week_of_year'] = missing_week
        
        # Manter colunas categóricas da semana anterior (ou próxima se anterior não existe)
        categorical_cols = [col for col in interpolated.columns 
                          if col not in ['pdv', 'internal_product_id', 'week_of_year', 'qty', 'gross'] 
                          and not col.endswith(('_prev', '_next'))]
        
        for col in categorical_cols:
            if col in interpolated.columns:
                interpolated[col] = interpolated[col].fillna(method='ffill').fillna(method='bfill')
        
        # Selecionar colunas finais
        final_cols = ['week_of_year', 'pdv', 'internal_product_id', 'qty', 'gross'] + categorical_cols
        interpolated_final = interpolated[final_cols].copy()
        
        # Adicionar dados interpolados ao DataFrame principal
        self.df_agg = pd.concat([self.df_agg, interpolated_final], ignore_index=True)
        
        print(f"   → Semana {missing_week} interpolada: {len(interpolated_final):,} registros adicionados")
    
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

        self.df_agg['rmean_4'] = grouped.shift(1).rolling(4, min_periods=4).mean()
        print("    → Média móvel 4 criada (min_periods=4)")
        
        self.df_agg['rstd_4'] = grouped.shift(1).rolling(4, min_periods=4).std()
        print("    → Desvio padrão móvel 4 criado (min_periods=4)")
        
        # Otimização para fração de não zeros
        is_nonzero = (self.df_agg['qty'] > 0).astype('int8')
        self.df_agg['nonzero_frac_8'] = (
            is_nonzero.groupby([self.df_agg['pdv'], self.df_agg['internal_product_id']])
            .shift(1)
            .rolling(8, min_periods=8)
            .mean()
        )
        print("    → Fração de não zeros nas últimas 8 criada")
        
        # Preencher NAs
        lag_cols = [c for c in self.df_agg.columns if c.startswith('lag_')] + ['rmean_4', 'rstd_4', 'nonzero_frac_8']
        self.df_agg[lag_cols] = self.df_agg[lag_cols].fillna(0)
        
        # Criar features categóricas
        self.create_categorical_features()
        
        # Definir colunas de features básicas (removido 'gross' para evitar data leakage)
        base_features = ['week_of_year'] + lag_cols
        
        # NÃO adicionar features de preço
        self.feature_columns = base_features + self.categorical_features
        
        print(f"   → {len(lag_cols)} features de lag criadas")
        print(f"   → Total de features: {len(self.feature_columns)}")
        return self.df_agg
    
    def create_categorical_features(self):
        """Cria features categóricas usando dummização e embeddings"""
        print("🎭 Criando features categóricas...")
        self.categorical_features = []
        
        # Verificar se as colunas categóricas existem nos dados
        available_dummy = [col for col in DUMMY_FEATURES if col in self.df_agg.columns]
        available_embedding = [col for col in EMBEDDING_FEATURES if col in self.df_agg.columns]
        
        if not available_dummy and not available_embedding:
            print("   → Nenhuma coluna categórica encontrada nos dados agregados")
            return

        # 1. Features Dummy (One-Hot Encoding)
        for col in available_dummy:
            print(f"  → Criando dummies para {col}...")
            # Usar apenas as top categorias mais frequentes (sem target info)
            unique_values = self.df_agg[col].value_counts().head(10).index.tolist()
            
            for value in unique_values:
                dummy_col = f'{col}_{value}'.replace(' ', '_').replace('-', '_')
                self.df_agg[dummy_col] = (self.df_agg[col] == value).astype('int8')
                self.categorical_features.append(dummy_col)
            
            print(f"    → {len(unique_values)} dummies criadas para {col}")
        
        # 2. Features de Embedding - REMOVENDO EMBEDDINGS SUPERVISIONADOS
        for col in available_embedding:
            print(f"  → Criando features simples para {col}...")
            
            # Substituir embeddings complexos por features simples baseadas apenas em frequência
            value_counts = self.df_agg[col].value_counts()
            
            # Feature 1: Log da frequência da categoria (sem target)
            freq_map = np.log1p(value_counts).to_dict()
            self.df_agg[f'{col}_log_freq'] = self.df_agg[col].map(freq_map).fillna(0)
            self.categorical_features.append(f'{col}_log_freq')
            
            # Feature 2: Ranking da categoria por frequência (sem target)
            rank_map = {v: i+1 for i, v in enumerate(value_counts.index)}
            self.df_agg[f'{col}_freq_rank'] = self.df_agg[col].map(rank_map).fillna(value_counts.shape[0]+1)
            self.categorical_features.append(f'{col}_freq_rank')
            
            print(f"    → 2 features simples criadas para {col} (log_freq + freq_rank)")

        print(f"   → Total de {len(self.categorical_features)} features categóricas criadas")

    def _estimate_memory_usage(self, n_unique, n_samples):
        """
        Estima uso de memória para one-hot encoding (método antigo)
        """
        # One-hot matrix: n_samples * n_unique * 4 bytes (float32)
        memory_bytes = n_samples * n_unique * 4
        memory_gb = memory_bytes / (1024**3)
        return memory_gb
    
    def _create_autoencoder_embeddings(self, categorical_series, target_series, n_unique, embedding_dim=10):
        """
        Cria embeddings usando Autoencoders para capturar padrões complexos nas categorias
        
        Args:
            categorical_series: Série categórica processada
            target_series: Série target para embeddings supervisionados
            n_unique: Número de categorias únicas
            embedding_dim: Dimensão do espaço de embedding
        
        Returns:
            embeddings: Array com embeddings para cada categoria
            autoencoder: Modelo treinado para aplicar em novos dados
        """
        if not TENSORFLOW_AVAILABLE:
            print("    → TensorFlow não disponível, usando PCA como fallback...")
            return self._create_pca_embeddings(categorical_series, n_unique, embedding_dim)
        
        print(f"    → Treinando Autoencoder OTIMIZADO para {n_unique} categorias...")
        
        # Pegar parâmetros de configuração
        ae_params = EMBEDDING_CONFIG.get('autoencoder_params', {})
        encoder_layers = ae_params.get('encoder_layers', [64, 32])  # Reduzido
        decoder_layers = ae_params.get('decoder_layers', [32, 64])  # Reduzido
        learning_rate = ae_params.get('learning_rate', 0.001)
        epochs = ae_params.get('epochs', 50)  # Reduzido
        batch_size = ae_params.get('batch_size', 64)  # Aumentado para eficiência
        validation_split = ae_params.get('validation_split', 0.15)  # Reduzido
        patience = ae_params.get('early_stopping_patience', 8)  # Reduzido
        
        # OTIMIZAÇÃO 1: Usar embeddings de entrada em vez de one-hot
        le = LabelEncoder()
        encoded_labels = le.fit_transform(categorical_series.fillna('outros'))
        
        # OTIMIZAÇÃO 2: Preparar features contextuais mais leves
        context_features = self._create_lightweight_context_features(categorical_series, target_series)
        
        # OTIMIZAÇÃO 3: Usar método ultra-eficiente para datasets grandes
        if n_unique > 200 or len(encoded_labels) > 100000:
            print(f"    → Dataset grande detectado, usando método ULTRA-EFICIENTE")
            category_embeddings = self._train_embeddings_efficient(
                encoded_labels, target_series, n_unique, embedding_dim,
                epochs, batch_size, learning_rate
            )
        else:
            # Método original otimizado para datasets menores
            category_embeddings = self._train_category_embeddings_batched(
                encoded_labels, context_features, n_unique, embedding_dim,
                epochs, batch_size, learning_rate, patience
            )
        
        # Mapear embeddings para cada registro
        embeddings = category_embeddings[encoded_labels]
        
        print(f"    → Autoencoder OTIMIZADO treinado com {len(category_embeddings)} embeddings únicos")
        print(f"    → Memória economizada: ~{self._estimate_memory_savings(n_unique, len(encoded_labels))}")
        
        # Criar encoder simples para novos dados
        encoder = self._create_simple_encoder(category_embeddings, le)
        
        return embeddings, encoder, le
    
    def _create_lightweight_context_features(self, categorical_series, target_series):
        """
        Cria features contextuais mais leves usando agregações eficientes
        """
        # Calcular estatísticas por categoria de forma mais eficiente
        df_temp = pd.DataFrame({
            'category': categorical_series,
            'target': target_series
        })
        
        # Usar apenas estatísticas essenciais para economizar memória
        category_stats = df_temp.groupby('category')['target'].agg([
            'mean', 'std', 'count'  # Reduzido de 6 para 3 estatísticas
        ]).fillna(0)
        
        # Normalizar para economizar espaço
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        category_stats_norm = scaler.fit_transform(category_stats.values)
        
        # Mapear de volta (mais eficiente que loop)
        category_to_idx = {cat: idx for idx, cat in enumerate(category_stats.index)}
        context_features = np.array([
            category_stats_norm[category_to_idx.get(cat, 0)] 
            for cat in categorical_series
        ])
        
        return context_features
    
    def _train_category_embeddings_batched(self, encoded_labels, context_features, n_unique, 
                                          embedding_dim, epochs, batch_size, learning_rate, patience):
        """
        Treina embeddings por categoria usando abordagem memory-efficient
        """
        # OTIMIZAÇÃO: Treinar apenas com representantes únicos de cada categoria
        unique_categories = np.unique(encoded_labels)
        
        # Para cada categoria, pegar uma amostra representativa
        category_samples = {}
        for cat in unique_categories:
            mask = encoded_labels == cat
            indices = np.where(mask)[0]
            
            # Pegar no máximo 100 exemplos por categoria para treino
            if len(indices) > 100:
                sample_indices = np.random.choice(indices, 100, replace=False)
            else:
                sample_indices = indices
            
            # Média das features contextuais para esta categoria
            avg_context = np.mean(context_features[sample_indices], axis=0)
            category_samples[cat] = avg_context
        
        print(f"    → Treinando com {len(category_samples)} amostras representativas")
        
        # Criar dados de treino compactos
        X_train = np.array(list(category_samples.values()))
        
        # Arquitetura simplificada para economizar memória
        input_dim = X_train.shape[1]  # 3 features contextuais
        
        # Modelo mais simples
        input_layer = Input(shape=(input_dim,))
        
        # Encoder mais leve
        x = Dense(32, activation='relu')(input_layer)  # Reduzido
        embedding_layer = Dense(embedding_dim, activation='linear', name='embedding')(x)
        
        # Decoder mais leve
        x = Dense(32, activation='relu')(embedding_layer)  # Reduzido
        output_layer = Dense(input_dim, activation='linear')(x)
        
        # Modelo
        autoencoder = Model(input_layer, output_layer)
        encoder = Model(input_layer, embedding_layer)
        
        # Compilar
        autoencoder.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse'
        )
        
        # Treinar com dados compactos
        early_stopping = EarlyStopping(
            monitor='loss',  # Sem validação para economizar memória
            patience=patience,
            restore_best_weights=True,
            verbose=0
        )
        
        # Treinar
        autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=min(batch_size, len(X_train)),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Gerar embeddings para todas as categorias
        category_embeddings = encoder.predict(X_train, verbose=0)
        
        # Criar lookup array
        embeddings_lookup = np.zeros((n_unique, embedding_dim))
        for i, cat in enumerate(category_samples.keys()):
            embeddings_lookup[cat] = category_embeddings[i]
        
        return embeddings_lookup
    
    def _train_embeddings_efficient(self, encoded_labels, target_series, n_unique, 
                                   embedding_dim, epochs, batch_size, learning_rate):
        """
        Método ultra-eficiente usando Embedding layers em vez de one-hot
        """
        print(f"    → Usando método ULTRA-EFICIENTE com Embedding layers")
        
        # Usar Embedding layer do TensorFlow (muito mais eficiente que one-hot)
        input_layer = Input(shape=(1,), dtype='int32')
        
        # Embedding layer - mapeia diretamente índices para vetores densos
        embedding_layer = Embedding(
            input_dim=n_unique,
            output_dim=embedding_dim,
            input_length=1
        )(input_layer)
        
        # Flatten para usar como features
        x = Flatten()(embedding_layer)
        
        # Predição do target (modelo supervisionado simples)
        target_pred = Dense(1, activation='linear')(x)
        
        # Modelo simples: apenas predição supervisionada
        model = Model(input_layer, target_pred)
        
        # Compilar para tarefa supervisionada
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse'
        )
        
        # Preparar dados
        X_input = encoded_labels.reshape(-1, 1)
        y_target = target_series.values.reshape(-1, 1)
        
        # Subsampling para treino eficiente
        if len(X_input) > 50000:  # Se muitos dados, fazer subsample
            indices = np.random.choice(len(X_input), 50000, replace=False)
            X_sample = X_input[indices]
            y_sample = y_target[indices]
        else:
            X_sample = X_input
            y_sample = y_target
        
        print(f"    → Treinando com {len(X_sample):,} amostras")
        
        # Treinar
        early_stopping = EarlyStopping(
            monitor='loss',
            patience=5,
            restore_best_weights=True,
            verbose=0
        )
        
        model.fit(
            X_sample, y_sample,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Criar modelo para extrair embeddings
        embedding_model = Model(input_layer, x)
        
        # Extrair embeddings para todas as categorias
        unique_cats = np.arange(n_unique).reshape(-1, 1)
        embeddings_per_category = embedding_model.predict(unique_cats, verbose=0)
        
        return embeddings_per_category
    
    def _create_simple_encoder(self, category_embeddings, le):
        """
        Cria um encoder simples que é apenas um lookup table
        """
        class SimpleEncoder:
            def __init__(self, embeddings_lookup, label_encoder):
                self.embeddings_lookup = embeddings_lookup
                self.label_encoder = label_encoder
            
            def predict(self, X_input, verbose=0):
                # X_input deve ser labels encoded
                return self.embeddings_lookup[X_input.astype(int)]
        
        return SimpleEncoder(category_embeddings, le)
    
    def _estimate_memory_savings(self, n_unique, n_samples):
        """
        Estima economia de memória comparando com one-hot encoding
        """
        # One-hot seria: n_samples * n_unique * 4 bytes (float32)
        onehot_size = n_samples * n_unique * 4 / (1024**3)  # GB
        
        # Método atual: lookup table pequena
        current_size = n_unique * 10 * 4 / (1024**3)  # GB (assumindo 10 dims)
        
        savings = onehot_size - current_size
        return f"{savings:.2f} GB economizados vs one-hot"
    
    def _create_context_features(self, categorical_series, target_series):
        """
        Cria features contextuais para melhorar os embeddings
        """
        # Features estatísticas por categoria
        category_stats = pd.DataFrame({
            'category': categorical_series,
            'target': target_series
        }).groupby('category')['target'].agg([
            'mean', 'std', 'count', 'median',
            lambda x: np.percentile(x, 25),  # Q1
            lambda x: np.percentile(x, 75),  # Q3
        ]).fillna(0)
        
        # Mapear de volta para cada registro
        context_features = []
        for cat in categorical_series:
            if cat in category_stats.index:
                context_features.append(category_stats.loc[cat].values)
            else:
                context_features.append(np.zeros(6))  # 6 estatísticas
        
        return np.array(context_features)
    
    def _create_pca_embeddings(self, categorical_series, n_unique, embedding_dim):
        """
        Método PCA como fallback quando TensorFlow não está disponível
        """
        # Criar matriz de embeddings simples (baseada em frequência e co-ocorrência)
        embedding_matrix = self._create_simple_embeddings(categorical_series, n_unique)
        
        # Aplicar PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(embedding_dim, n_unique-1))
        reduced_embeddings = pca.fit_transform(embedding_matrix)
        
        # Mapear de volta para os dados
        le = LabelEncoder()
        encoded_values = le.fit_transform(categorical_series.fillna('outros'))
        embeddings = reduced_embeddings[encoded_values]
        
        return embeddings, pca, le
    
    def apply_autoencoder_embeddings(self, data, col):
        """
        Aplica embeddings de autoencoder treinados a novos dados
        
        Args:
            data: DataFrame com novos dados
            col: Nome da coluna categórica
        
        Returns:
            DataFrame com embeddings aplicados
        """
        if col not in self.autoencoder_transformers:
            print(f"⚠️ Autoencoder para {col} não encontrado, pulando...")
            return data
        
        if not TENSORFLOW_AVAILABLE:
            print(f"⚠️ TensorFlow não disponível para aplicar autoencoder em {col}")
            return data
        
        encoder = self.autoencoder_transformers[col]
        le = self.label_encoders[col]
        
        # Processar dados da mesma forma que no treinamento
        processed_col = data[col].apply(
            lambda x: x if x in le.classes_ else 'outros'
        )
        
        # Label encoding
        try:
            encoded_labels = le.transform(processed_col.fillna('outros'))
        except ValueError:
            # Se houver categorias não vistas, mapear para 'outros'
            encoded_labels = []
            for val in processed_col.fillna('outros'):
                if val in le.classes_:
                    encoded_labels.append(le.transform([val])[0])
                else:
                    encoded_labels.append(le.transform(['outros'])[0])
            encoded_labels = np.array(encoded_labels)
        
        # One-hot encoding
        n_unique = len(le.classes_)
        one_hot_input = to_categorical(encoded_labels, num_classes=n_unique)
        
        # Features contextuais (usar médias de treino como fallback)
        context_features = self._create_context_features_inference(processed_col, data.get('qty', None))
        
        # Combinar dados
        X_input = np.concatenate([one_hot_input, context_features], axis=1)
        
        # Aplicar encoder
        embeddings = encoder.predict(X_input, verbose=0)
        
        # Adicionar embeddings ao DataFrame
        for i in range(embeddings.shape[1]):
            data[f'{col}_ae_{i}'] = embeddings[:, i]
        
        return data
    
    def _create_context_features_inference(self, categorical_series, target_series=None):
        """
        Cria features contextuais para inferência (sem vazamento de dados)
        """
        if target_series is None:
            # Usar estatísticas globais como fallback
            context_features = np.zeros((len(categorical_series), 6))
        else:
            context_features = self._create_context_features(categorical_series, target_series)
        
        return context_features

    def _create_simple_embeddings(self, categorical_series, n_unique):
        """Cria embeddings simples baseados em frequência e co-ocorrência"""
        # Matriz simples: cada linha é uma categoria, colunas são features estatísticas
        embedding_dim = min(20, n_unique)
        embedding_matrix = np.zeros((n_unique, embedding_dim))
        
        value_counts = categorical_series.value_counts()
        unique_values = categorical_series.unique()
        
        for i, value in enumerate(unique_values):
            # Feature 1: Log da frequência
            embedding_matrix[i, 0] = np.log1p(value_counts.get(value, 1))
            
            # Features adicionais: estatísticas baseadas em posição e contexto
            if embedding_dim > 1:
                # Usar hash simples para criar features determinísticas
                hash_val = hash(str(value)) % 1000000
                for j in range(1, min(embedding_dim, 10)):
                    embedding_matrix[i, j] = (hash_val % (j * 100 + 1)) / (j * 100 + 1)
        
        return embedding_matrix

    def create_unit_price_features(self):
        """
        Cria features de preço unitário baseado no gross_value e quantity
        Implementa previsão de preços futuros baseada na tendência histórica
        """
        print("💰 Criando features de preço unitário...")
        
        # Verificar se as colunas necessárias existem
        if 'gross' not in self.df_agg.columns:
            print("   → ⚠️ Coluna 'gross' não encontrada. Pulando criação de features de preço.")
            return
        
        # Calcular preço unitário onde faz sentido (qty > 0 e gross > 0)
        valid_mask = (self.df_agg['qty'] > 0) & (self.df_agg['gross'] > 0)
        
        # Inicializar coluna de preço unitário
        self.df_agg['unit_price'] = 0.0
        
        # Calcular preço unitário e arredondar para 2 casas decimais
        self.df_agg.loc[valid_mask, 'unit_price'] = (
            self.df_agg.loc[valid_mask, 'gross'] / self.df_agg.loc[valid_mask, 'qty']
        ).round(2)
        
        print(f"   → Preço unitário calculado para {valid_mask.sum():,} registros")
        
        # Criar features de lag do preço unitário
        self.df_agg = self.df_agg.sort_values(['pdv', 'internal_product_id', 'week_of_year'])
        
        # Lags de preço unitário
        price_lags = [1, 2, 4]
        for lag in price_lags:
            col_name = f'unit_price_lag_{lag}'
            self.df_agg[col_name] = (
                self.df_agg.groupby(['pdv', 'internal_product_id'])['unit_price']
                .shift(lag)
                .fillna(0)
                .round(2)
            )
            print(f"   → Feature {col_name} criada")
        
        # Feature de variação de preço (diferença percentual)
        self.df_agg['unit_price_change_pct'] = (
            self.df_agg.groupby(['pdv', 'internal_product_id'])['unit_price']
            .pct_change()
            .fillna(0)
            .round(4)
        )
        print("   → Feature unit_price_change_pct criada")
        
        # Feature de preço médio das últimas 4 semanas
        self.df_agg['unit_price_rolling_mean_4'] = (
            self.df_agg.groupby(['pdv', 'internal_product_id'])['unit_price']
            .shift(1)
            .rolling(4, min_periods=1)
            .mean()
            .fillna(0)
            .round(2)
        )
        print("   → Feature unit_price_rolling_mean_4 criada")
        
        # Criar previsões de preço para dados futuros (janeiro 2023)
        self._create_future_price_predictions()
        
        print(f"   → Features de preço unitário criadas com sucesso")
    
    def _create_future_price_predictions(self):
        """
        Cria previsões de preço unitário para dados futuros usando tendência histórica
        """
        print("   → Criando previsões de preço para dados futuros...")
        
        # Identificar combinações PDV-produto com dados históricos suficientes
        price_evolution = self.df_agg[self.df_agg['unit_price'] > 0].copy()
        
        if len(price_evolution) == 0:
            print("   → ⚠️ Nenhum dado de preço válido encontrado")
            return
        
        # Agrupar por PDV-produto para calcular tendências
        predicted_prices = []
        
        for (pdv, product_id), group in price_evolution.groupby(['pdv', 'internal_product_id']):
            group_sorted = group.sort_values('week_of_year')
            
            if len(group_sorted) >= 2:
                # Usar os últimos 6 registros ou todos disponíveis
                recent_data = group_sorted.tail(6)
                weeks = np.arange(len(recent_data))
                prices = recent_data['unit_price'].values
                
                if len(prices) > 1 and np.any(prices > 0):
                    # Calcular tendência linear
                    slope, intercept = np.polyfit(weeks, prices, 1)
                    last_price = prices[-1]
                    
                    # Prever preço para próximo período
                    predicted_price = last_price + slope
                    
                    # Aplicar limites realistas
                    min_price = last_price * 0.8  # Máx 20% redução
                    max_price = last_price * 1.3  # Máx 30% aumento
                    predicted_price = np.clip(predicted_price, min_price, max_price)
                else:
                    predicted_price = group_sorted['unit_price'].iloc[-1]
            else:
                predicted_price = group_sorted['unit_price'].iloc[-1]
            
            predicted_prices.append({
                'pdv': pdv,
                'internal_product_id': product_id,
                'predicted_unit_price': round(predicted_price, 2)
            })
        
        # Converter para DataFrame para merge
        predicted_df = pd.DataFrame(predicted_prices)
        
        # Adicionar coluna de preço previsto ao dataset principal
        self.df_agg = self.df_agg.merge(
            predicted_df, 
            on=['pdv', 'internal_product_id'], 
            how='left'
        )
        
        # Preencher valores ausentes com o último preço observado
        self.df_agg['predicted_unit_price'] = self.df_agg['predicted_unit_price'].fillna(
            self.df_agg['unit_price']
        ).round(2)
        
        # Renomear para feature final
        self.df_agg['unit_price_predicted'] = self.df_agg['predicted_unit_price']
        self.df_agg = self.df_agg.drop(columns=['predicted_unit_price'], errors='ignore')
        
        print(f"   → Previsões de preço criadas para {len(predicted_df):,} combinações PDV-produto")

    def detect_and_treat_outliers(self, rolling_window=12, z_threshold=5.0, treatment='winsorize', 
                                  adaptive_threshold=True, min_observations=10, absolute_cap_multiplier=3,
                                  method='median_imputation', percentile_cap=99.5, absolute_cap=220):
        """
        Detecta e trata outliers usando diferentes métodos
        
        Args:
            method (str): 'percentile_cap', 'z_score', 'median_imputation', ou 'hybrid'
            percentile_cap (float): Percentil para cap (ex: 95.0)
            absolute_cap (float): Cap absoluto como backup
            rolling_window (int): Tamanho da janela rolling para calcular média e desvio
            z_threshold (float): Threshold base do Z-score para considerar outlier
            treatment (str): Tipo de tratamento ('winsorize', 'remove', 'cap', 'median_impute')
            adaptive_threshold (bool): Se True, ajusta threshold baseado nas características do grupo
            min_observations (int): Mínimo de observações para calcular estatísticas confiáveis
            absolute_cap_multiplier (float): Cap absoluto como múltiplo da média do grupo (método antigo)
        """
        
        if method == 'median_imputation':
            return self._median_imputation_outliers(percentile_cap, absolute_cap)
        elif method == 'percentile_cap':
            return self._percentile_cap_outliers(percentile_cap, absolute_cap)
        else:
            return self._detect_and_treat_outliers(rolling_window, z_threshold, treatment, 
                                                               adaptive_threshold, min_observations, absolute_cap_multiplier)
    
    def _median_imputation_outliers(self, percentile_cap=95.0, absolute_cap=24, rolling_window=12):
        """
        Método de imputação por mediana: substitui outliers pela mediana usando apenas dados passados
        """
        print(f"🎯 Aplicando imputacao por mediana para outliers...")
        print(f"   → Metodo: Detectar outliers usando janela rolling de {rolling_window} semanas")
        print(f"   → Percentil: {percentile_cap}% | Cap absoluto: {absolute_cap}")
        
        # Ordenar dados por PDV, produto e semana
        self.df_agg = self.df_agg.sort_values(['pdv', 'internal_product_id', 'week_of_year']).reset_index(drop=True)
        
        # Calcular estatísticas antes
        total_records = len(self.df_agg)
        total_volume_before = self.df_agg['qty'].sum()
        
        print(f"   → Processando {total_records:,} registros de forma vetorizada...")
        
        # Usar operações vetorizadas para calcular rolling statistics com shift=1 (sem data leakage)
        grouped = self.df_agg.groupby(['pdv', 'internal_product_id'])['qty']
        
        # Calcular estatísticas rolling usando apenas dados passados (shift=1)
        rolling_median = grouped.shift(1).rolling(rolling_window, min_periods=3).median()
        rolling_percentile = grouped.shift(1).rolling(rolling_window, min_periods=3).quantile(percentile_cap / 100)
        
        # Aplicar cap absoluto e percentil (usar o menor)
        outlier_threshold = np.minimum(rolling_percentile.fillna(absolute_cap), absolute_cap)
        
        # Identificar outliers de forma vetorizada
        outliers_mask = (self.df_agg['qty'] > outlier_threshold) & (outlier_threshold.notna())
        
        # Contar outliers
        outliers_count = outliers_mask.sum()
        outliers_volume = self.df_agg.loc[outliers_mask, 'qty'].sum()
        
        # Imputar outliers com a mediana rolling (apenas dados passados)
        self.df_agg.loc[outliers_mask, 'qty'] = rolling_median.loc[outliers_mask]
        
        # Para casos onde rolling_median é NaN, usar valor conservador (mínimo entre 1 e valor original)
        still_nan_mask = outliers_mask & self.df_agg['qty'].isna()
        if still_nan_mask.any():
            self.df_agg.loc[still_nan_mask, 'qty'] = 1.0  # Valor conservador para casos extremos
        
        # Estatísticas após tratamento
        total_volume_after = self.df_agg['qty'].sum()
        volume_reduction = ((total_volume_before - total_volume_after) / total_volume_before) * 100
        
        # Calcular estatísticas para o relatório
        final_median = self.df_agg['qty'].median()
        final_max = self.df_agg['qty'].max()
        final_mean = self.df_agg['qty'].mean()
        
        print(f"\n📊 RESULTADO DA IMPUTAÇÃO POR MEDIANA:")
        print(f"   → Registros afetados: {outliers_count:,} ({outliers_count/total_records*100:.3f}%)")
        print(f"   → Volume antes: {total_volume_before:,.0f}")
        print(f"   → Volume depois: {total_volume_after:,.0f}")
        print(f"   → Redução de volume: {volume_reduction:.1f}%")
        print(f"   → Novo máximo: {final_max:.1f}")
        print(f"   → Nova média: {final_mean:.2f}")
        print(f"   → Nova mediana: {final_median:.1f}")
        
        # Criar estatísticas para compatibilidade
        self.outlier_stats = pd.DataFrame({
            'method': ['median_imputation_no_leakage'],
            'total_outliers': [outliers_count],
            'volume_reduction_pct': [volume_reduction],
            'threshold_used': [absolute_cap],  # Cap absoluto como referência
            'final_median': [final_median]
        })
        
        print("✅ Imputação por mediana concluída!")
        return self.df_agg
    
    def _percentile_cap_outliers(self, percentile_cap=99.5, absolute_cap=220, rolling_window=12):
        """
        Método mais robusto OTIMIZADO: cap baseado em percentil usando apenas dados passados
        """
        print(f"🎯 Aplicando tratamento robusto de outliers...")
        print(f"   → Metodo: Cap baseado em percentil {percentile_cap}% + cap absoluto {absolute_cap}")
        print(f"   → Janela rolling: {rolling_window} semanas")
        
        # Ordenar dados por PDV, produto e semana
        self.df_agg = self.df_agg.sort_values(['pdv', 'internal_product_id', 'week_of_year']).reset_index(drop=True)
        
        # Calcular estatísticas antes
        total_records = len(self.df_agg)
        total_volume_before = self.df_agg['qty'].sum()
        
        print(f"   → Processando {total_records:,} registros de forma vetorizada...")
        
        # Usar operações vetorizadas para calcular rolling statistics com shift=1
        grouped = self.df_agg.groupby(['pdv', 'internal_product_id'])['qty']
        
        # Calcular cap rolling usando apenas dados passados (shift=1)
        rolling_percentile = grouped.shift(1).rolling(rolling_window, min_periods=3).quantile(percentile_cap / 100)
        
        # Aplicar cap absoluto e percentil (usar o menor)
        final_cap = np.minimum(rolling_percentile.fillna(absolute_cap), absolute_cap)
        
        # Identificar outliers de forma vetorizada
        outliers_mask = (self.df_agg['qty'] > final_cap) & (final_cap.notna())
        
        # Contar outliers
        outliers_count = outliers_mask.sum()
        outliers_volume = (self.df_agg.loc[outliers_mask, 'qty'] - final_cap.loc[outliers_mask]).sum()
        
        # Aplicar cap de forma vetorizada
        self.df_agg.loc[outliers_mask, 'qty'] = final_cap.loc[outliers_mask]
        
        # Estatísticas após tratamento
        total_volume_after = self.df_agg['qty'].sum()
        volume_reduction = ((total_volume_before - total_volume_after) / total_volume_before) * 100
        
        # Calcular estatísticas finais
        final_max = self.df_agg['qty'].max()
        final_mean = self.df_agg['qty'].mean()
        
        print(f"\n📊 RESULTADO DO TRATAMENTO ROBUSTO:")
        print(f"   → Registros afetados: {outliers_count:,} ({outliers_count/total_records*100:.3f}%)")
        print(f"   → Volume antes: {total_volume_before:,.0f}")
        print(f"   → Volume depois: {total_volume_after:,.0f}")
        print(f"   → Redução de volume: {volume_reduction:.1f}%")
        print(f"   → Novo máximo: {final_max:.1f}")
        print(f"   → Nova média: {final_mean:.2f}")
        
        # Criar estatísticas para compatibilidade
        self.outlier_stats = pd.DataFrame({
            'method': ['percentile_cap_no_leakage'],
            'total_outliers': [outliers_count],
            'volume_reduction_pct': [volume_reduction],
            'final_cap': [absolute_cap]  # Cap absoluto como referência
        })
        
        print("✅ Tratamento robusto de outliers concluído!")
        return self.df_agg

    def _detect_and_treat_outliers(self, rolling_window=12, z_threshold=5.0, 
                                               treatment='winsorize', adaptive_threshold=True, 
                                               min_observations=10, absolute_cap_multiplier=3.0):
        """
        Versão otimizada para detecção e tratamento de outliers por PDV-produto.
        """
        print(f"🎯 Detectando outliers com processamento paralelo...")
        print(f"   → Janela rolling: {rolling_window} semanas | Z-score base: {z_threshold}")
        start_time = pd.Timestamp.now()
        
        self.df_agg = self.df_agg.sort_values(['pdv', 'internal_product_id', 'week_of_year'])
        
        # Criar lista de grupos para processamento paralelo
        groups_list = [(name, group) for name, group in self.df_agg.groupby(['pdv', 'internal_product_id'], sort=False)]
        n_groups = len(groups_list)
        print(f"  → Processando {n_groups:,} combinações PDV-produto em paralelo...")

        # Parâmetros para outliers
        outlier_params = {
            'rolling_window': rolling_window,
            'z_threshold': z_threshold,
            'treatment': treatment,
            'adaptive_threshold': adaptive_threshold,
            'min_observations': min_observations,
            'absolute_cap_multiplier': absolute_cap_multiplier
        }

        # Criar argumentos para a função wrapper
        args_list = [(group_tuple, outlier_params) for group_tuple in groups_list]

        # Determinar número de processos otimizado
        num_processes = min(mp.cpu_count(), n_groups)  # Não criar mais processos que grupos
        print(f"  → Utilizando {num_processes} processos...")

        # Usar chunksize otimizado para melhor balanceamento
        chunksize = max(1, n_groups // (num_processes * 4))  # 4 chunks por processo
        
        results = []
        # Usar Pool para processamento paralelo com barra de progresso
        with mp.Pool(processes=num_processes) as pool:
            with tqdm(total=n_groups, desc="Analisando grupos PDV-produto") as pbar:
                # Usar imap com chunksize para melhor performance
                for result in pool.imap(_process_outlier_group_wrapper, args_list, chunksize=chunksize):
                    results.append(result)
                    pbar.update()

        # Coletar resultados de forma mais eficiente
        processed_groups = []
        group_stats = []
        
        for res in results:
            if res and len(res) == 2:
                group_data, stats = res
                if group_data is not None and not group_data.empty:
                    processed_groups.append(group_data)
                if stats:
                    group_stats.append(stats)

        # Reconstruir o DataFrame de forma otimizada
        if processed_groups:
            # Usar concat com ignore_index=True e sort=False para melhor performance
            self.df_agg = pd.concat(processed_groups, ignore_index=True, sort=False)
        
        # Criar DataFrame de estatísticas de forma mais eficiente
        if group_stats:
            stats_df = pd.DataFrame(group_stats)
            total_outliers = stats_df['n_outliers'].sum()
        else:
            stats_df = pd.DataFrame()
            total_outliers = 0
        
        # Estatísticas finais
        end_time = pd.Timestamp.now()
        duration = (end_time - start_time).total_seconds()
        
        # A contagem de outlier_pct pode ser imprecisa se 'remove' for usado
        # pois o len(self.df_agg) mudou. Calculamos antes da remoção.
        initial_records = sum(stats_df['n_obs']) if not stats_df.empty else 1
        outlier_pct = (total_outliers / initial_records) * 100 if initial_records > 0 else 0
        
        print(f"\n📊 RELATÓRIO DE OUTLIERS PERSONALIZADOS:")
        print(f"   → Total de outliers: {total_outliers:,} ({outlier_pct:.2f}% dos dados originais)")
        print(f"   → Grupos processados: {len(stats_df):,}")
        print(f"   → Tratamento aplicado: {treatment}")
        
        if not stats_df.empty:
            print(f"\n📈 ESTATÍSTICAS POR GRUPO:")
            print(f"   → Threshold médio usado: {stats_df['threshold_used'].mean():.2f}")
            print(f"   → Grupos com outliers: {(stats_df['n_outliers'] > 0).sum():,}")
            print(f"   → CV médio: {stats_df['cv'].mean():.2f}")
            print(f"   → % zero média: {stats_df['zero_pct'].mean()*100:.1f}%")
            
            top_outliers = stats_df.nlargest(5, 'outlier_pct')
            if not top_outliers.empty:
                print(f"\n🎯 TOP 5 GRUPOS COM MAIS OUTLIERS:")
                for _, row in top_outliers.iterrows():
                    print(f"   → PDV {row['pdv']}, Produto {row['product']}: "
                          f"{row['n_outliers']} outliers ({row['outlier_pct']:.1f}%)")
        
        print(f"\n✅ Processamento paralelo concluído!")
        print(f"   → Tempo total: {duration:.2f} segundos")
        print(f"   → Dados finais: {len(self.df_agg):,} registros")
        
        self.outlier_stats = stats_df
        
        return self.df_agg
    
    def analyze_outlier_treatment(self):
        """
        Analisa os resultados do tratamento personalizado de outliers
        Retorna insights sobre os padrões encontrados
        """
        if not hasattr(self, 'outlier_stats') or self.outlier_stats is None:
            print("⚠️ Nenhuma estatística de outliers disponível. Execute detect_and_treat_outliers primeiro.")
            return None
        
        stats = self.outlier_stats
        
        # Verificar se o DataFrame está vazio ou não tem as colunas necessárias
        if stats.empty:
            print("📊 ANÁLISE DETALHADA DO TRATAMENTO DE OUTLIERS:")
            print("="*60)
            print("🎯 RESUMO GERAL:")
            print("   → Nenhum grupo foi processado ou nenhuma estatística foi coletada.")
            print("   → Verifique se os dados de entrada estão corretos.")
            return {
                'total_groups': 0,
                'groups_with_outliers': 0,
                'total_outliers': 0,
                'stats_by_pdv': None,
                'stats_by_product': None,
                'threshold_stats': None,
                'detailed_stats': stats
            }
        
        # Verificar se é o novo método median_imputation
        if 'method' in stats.columns and not stats.empty:
            method_name = stats['method'].iloc[0]
            
            if method_name in ['median_imputation']:
                print("📊 ANÁLISE DETALHADA DO TRATAMENTO DE OUTLIERS:")
                print("="*60)
                print(f"✅ Método aplicado: {method_name}")
                print(f"   → Total de outliers tratados: {stats['total_outliers'].iloc[0]:,}")
                print(f"   → Redução de volume: {stats['volume_reduction_pct'].iloc[0]:.1f}%")
                
                if 'final_median' in stats.columns:
                    print(f"   → Mediana final: {stats['final_median'].iloc[0]:.1f}")
                elif 'median_value' in stats.columns:
                    print(f"   → Valor da mediana: {stats['median_value'].iloc[0]:.1f}")
                    
                print("   → Tratamento por imputação com mediana - muito conservador!")
                return {
                    'total_groups': 1,
                    'groups_with_outliers': 1 if stats['total_outliers'].iloc[0] > 0 else 0,
                    'total_outliers': stats['total_outliers'].iloc[0],
                    'method': method_name,
                    'detailed_stats': stats
                }
            
            elif method_name in ['percentile_cap']:
                print("📊 ANÁLISE DETALHADA DO TRATAMENTO DE OUTLIERS:")
                print("="*60)
                print(f"✅ Método aplicado: {method_name}")
                print(f"   → Total de outliers tratados: {stats['total_outliers'].iloc[0]:,}")
                print(f"   → Redução de volume: {stats['volume_reduction_pct'].iloc[0]:.1f}%")
                print(f"   → Cap aplicado: {stats['final_cap'].iloc[0]:.0f}")
                    
                print("   → Tratamento baseado em percentil - mais robusto!")
                return {
                    'total_groups': 1,
                    'groups_with_outliers': 1 if stats['total_outliers'].iloc[0] > 0 else 0,
                    'total_outliers': stats['total_outliers'].iloc[0],
                    'method': method_name,
                    'detailed_stats': stats
                }
        
        # Verificar se as colunas necessárias existem (método antigo)
        required_cols = ['n_outliers', 'pdv', 'product', 'cv', 'zero_pct', 'mean_qty', 'threshold_used']
        missing_cols = [col for col in required_cols if col not in stats.columns]
        if missing_cols:
            print("📊 ANÁLISE DETALHADA DO TRATAMENTO DE OUTLIERS:")
            print("="*60)
            print(f"⚠️ Colunas necessárias não encontradas: {missing_cols}")
            print("   → O processamento pode ter falhado. Verifique os logs anteriores.")
            return {
                'total_groups': len(stats),
                'groups_with_outliers': 0,
                'total_outliers': 0,
                'stats_by_pdv': None,
                'stats_by_product': None,
                'threshold_stats': None,
                'detailed_stats': stats
            }
        
        print("📊 ANÁLISE DETALHADA DO TRATAMENTO DE OUTLIERS:")
        print("="*60)
        
        # Análise geral
        total_groups = len(stats)
        groups_with_outliers = (stats['n_outliers'] > 0).sum()
        
        print(f"🎯 RESUMO GERAL:")
        print(f"   → Total de grupos PDV-produto: {total_groups:,}")
        print(f"   → Grupos com outliers: {groups_with_outliers:,} ({groups_with_outliers/total_groups*100:.1f}%)")
        print(f"   → Total de outliers encontrados: {stats['n_outliers'].sum():,}")
        
        # Análise por características dos produtos
        print(f"\n📈 ANÁLISE POR CARACTERÍSTICAS:")
        
        # Produtos por faixa de variabilidade (CV)
        cv_stats = stats.dropna(subset=['cv'])
        if len(cv_stats) > 0:
            print(f"   → Coeficiente de Variação:")
            print(f"     • Baixa variabilidade (CV < 1.0): {(cv_stats['cv'] < 1.0).sum():,} grupos")
            print(f"     • Média variabilidade (1.0 ≤ CV < 2.0): {((cv_stats['cv'] >= 1.0) & (cv_stats['cv'] < 2.0)).sum():,} grupos")
            print(f"     • Alta variabilidade (CV ≥ 2.0): {(cv_stats['cv'] >= 2.0).sum():,} grupos")
        
        # Produtos por % de zeros
        print(f"   → Produtos com muitos zeros:")
        print(f"     • Até 30% zeros: {(stats['zero_pct'] <= 0.3).sum():,} grupos")
        print(f"     • 30-70% zeros: {((stats['zero_pct'] > 0.3) & (stats['zero_pct'] <= 0.7)).sum():,} grupos")
        print(f"     • Mais de 70% zeros: {(stats['zero_pct'] > 0.7).sum():,} grupos")
        
        # Produtos por volume médio
        print(f"   → Produtos por volume médio:")
        print(f"     • Baixo volume (< 10): {(stats['mean_qty'] < 10).sum():,} grupos")
        print(f"     • Médio volume (10-100): {((stats['mean_qty'] >= 10) & (stats['mean_qty'] < 100)).sum():,} grupos")
        print(f"     • Alto volume (≥ 100): {(stats['mean_qty'] >= 100).sum():,} grupos")
        
        # Top PDVs com mais outliers (apenas se há outliers)
        if groups_with_outliers > 0:
            print(f"\n🏪 TOP 10 PDVs COM MAIS OUTLIERS:")
            pdv_outliers = stats.groupby('pdv').agg({
                'n_outliers': 'sum',
                'product': 'count'
            }).sort_values('n_outliers', ascending=False).head(10)
            
            for pdv, row in pdv_outliers.iterrows():
                if row['n_outliers'] > 0:  # Só mostrar PDVs que realmente têm outliers
                    print(f"   → PDV {pdv}: {row['n_outliers']:,} outliers em {row['product']:,} produtos")
            
            # Top produtos com mais outliers
            print(f"\n📦 TOP 10 PRODUTOS COM MAIS OUTLIERS:")
            product_outliers = stats.groupby('product').agg({
                'n_outliers': 'sum',
                'pdv': 'count'
            }).sort_values('n_outliers', ascending=False).head(10)
            
            for product, row in product_outliers.iterrows():
                if row['n_outliers'] > 0:  # Só mostrar produtos que realmente têm outliers
                    print(f"   → Produto {product}: {row['n_outliers']:,} outliers em {row['pdv']:,} PDVs")
        else:
            print(f"\n✅ NENHUM OUTLIER DETECTADO:")
            print(f"   → Todos os {total_groups:,} grupos PDV-produto estão dentro dos parâmetros normais")
            print(f"   → Os dados parecem estar bem comportados ou os thresholds são muito altos")
            pdv_outliers = None
            product_outliers = None
        
        # Thresholds utilizados
        print(f"\n🎛️ THRESHOLDS ADAPTATIVOS UTILIZADOS:")
        threshold_stats = stats['threshold_used'].describe()
        print(f"   → Mínimo: {threshold_stats['min']:.2f}")
        print(f"   → Médio: {threshold_stats['mean']:.2f}")
        print(f"   → Máximo: {threshold_stats['max']:.2f}")
        print(f"   → Desvio padrão: {threshold_stats['std']:.2f}")
        
        return {
            'total_groups': total_groups,
            'groups_with_outliers': groups_with_outliers,
            'total_outliers': stats['n_outliers'].sum(),
            'stats_by_pdv': pdv_outliers,
            'stats_by_product': product_outliers,
            'threshold_stats': threshold_stats,
            'detailed_stats': stats
        }
    
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

def process_data(file_path, treat_outliers=True, outlier_params=None):
    """
    Função principal para processar dados completos
    
    Args:
        file_path (str): Caminho para o arquivo de dados
        treat_outliers (bool): Se deve aplicar tratamento de outliers
        outlier_params (dict): Parâmetros para tratamento de outliers personalizado
    """
    if outlier_params is None:
        outlier_params = OUTLIER_PARAMS
    
    processor = DataProcessor()
    
    # Pipeline de processamento
    processor.load_data(file_path)
    processor.aggregate_data()
    
    processor.df_agg = processor.df_agg[processor.df_agg['week_of_year'] != 53]

    # Salvar dados antes do tratamento de outliers para comparação
    df_before_outliers = processor.df_agg.copy() if treat_outliers else None
    
    # Tratamento de outliers personalizado
    if treat_outliers:
        processor.detect_and_treat_outliers(**outlier_params)
        # Análise dos resultados
        outlier_analysis = processor.analyze_outlier_treatment()
        
        # Gerar gráficos de comparação antes/depois do tratamento
        from .visualization import DataVisualizer
        visualizer = DataVisualizer()
        print("\n🎨 Gerando visualizações de comparação do tratamento de outliers...")
        outlier_stats = getattr(processor, 'outlier_stats', None)
        visualizer.plot_outlier_treatment_comparison(
            df_before_outliers, 
            processor.df_agg, 
            outlier_stats=outlier_stats,
            save_plots=True
        )
    else:
        outlier_analysis = None
    
    processor.create_lag_features()

    # Dividir dados
    from .config import TRAIN_CUTOFF, VALID_START, VALID_END
    train_data, valid_data = processor.split_train_validation(train_cutoff=TRAIN_CUTOFF, 
                                                             valid_start=VALID_START, 
                                                             valid_end=VALID_END)

    # Preparar features e targets
    X_train, y_train = processor.prepare_features_target(train_data)
    X_val, y_val = processor.prepare_features_target(valid_data)
    
    # SALVAR DADOS PARA ANÁLISE ANTES DO TREINAMENTO
    print("💾 Salvando dados de treinamento e validação para análise...")
    
    # Salvar dados de treinamento
    train_sample = pd.concat([X_train, y_train], axis=1)
    train_sample.to_csv('data_for_model_training.csv', index=False, encoding='utf-8')
    print(f"   → Dados de treinamento salvos: {len(train_sample):,} registros")
    print(f"   → Features: {len(X_train.columns)} colunas")
    print(f"   → Arquivo: data_for_model_training.csv")
    
    # Salvar dados de validação
    val_sample = pd.concat([X_val, y_val], axis=1)
    val_sample.to_csv('data_for_model_validation.csv', index=False, encoding='utf-8')
    print(f"   → Dados de validação salvos: {len(val_sample):,} registros")
    print(f"   → Arquivo: data_for_model_validation.csv")
    
    # Salvar informações das features
    feature_info = pd.DataFrame({
        'feature_name': processor.feature_columns,
        'feature_type': ['categorical' if col.startswith(('categoria_', 'premise_')) or col.endswith(('_target_enc', '_emb_0', '_emb_1', '_emb_2', '_emb_3', '_emb_4', '_emb_5', '_emb_6', '_emb_7', '_emb_8', '_emb_9')) else 'numerical' for col in processor.feature_columns]
    })
    feature_info.to_csv('model_features_info.csv', index=False, encoding='utf-8')
    print(f"   → Informações das features salvas: model_features_info.csv")
    
    # Salvar estatísticas descritivas das features
    features_stats = X_train.describe().T
    features_stats['feature_name'] = features_stats.index
    features_stats = features_stats[['feature_name', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    features_stats.to_csv('model_features_statistics.csv', index=False, encoding='utf-8')
    print(f"   → Estatísticas das features salvas: model_features_statistics.csv")
    
    # Estatísticas
    stats = processor.get_basic_stats()
    
    return {
        'processor': processor,
        'raw_data': processor.df_raw,
        'aggregated_data': processor.df_agg,
        'train_data': (X_train, y_train),
        'validation_data': (X_val, y_val),
        'feature_columns': processor.feature_columns,
        'stats': stats,
        'outlier_analysis': outlier_analysis
    }