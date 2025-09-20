# Módulo para tratamento e processamento de dados
import pandas as pd
import numpy as np
from tqdm import tqdm
from .config import LAG_FEATURES, OUTLIER_PARAMS
import warnings
import multiprocessing as mp
from functools import partial
warnings.filterwarnings('ignore')

def _process_outlier_group_wrapper(args):
    """Wrapper function para desempacotar argumentos do multiprocessing."""
    name_group_tuple, outlier_params = args
    name, group = name_group_tuple
    group.name = name
    return _process_outlier_group(group, **outlier_params)

def _process_outlier_group(group, rolling_window, z_threshold, treatment, adaptive_threshold, min_observations):
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
            # Método rolling otimizado usando numpy
            group_sorted = group.sort_values('week_of_year')
            qty_sorted = group_sorted['qty'].values
            
            # Calcular rolling statistics de forma vetorizada
            rolling_mean = np.full(n_obs, mean_qty)
            rolling_std = np.full(n_obs, std_qty)
            
            for i in range(1, min(n_obs, rolling_window + 1)):
                if i >= 3:  # min_periods
                    rolling_mean[i] = np.mean(qty_sorted[:i])
                    rolling_std[i] = np.std(qty_sorted[:i]) or std_qty
            
            for i in range(rolling_window + 1, n_obs):
                rolling_mean[i] = np.mean(qty_sorted[i-rolling_window:i])
                rolling_std[i] = np.std(qty_sorted[i-rolling_window:i]) or std_qty
            
            # Evitar divisão por zero
            rolling_std = np.where(rolling_std == 0, std_qty, rolling_std)
            
            # Calcular Z-scores
            z_scores = np.abs((qty_sorted - rolling_mean) / rolling_std)
            outliers_mask = z_scores > threshold
            
            # Reordenar mask para corresponder ao grupo original
            if not group_sorted.index.equals(group.index):
                sort_indices = group_sorted.index.get_indexer(group.index)
                outliers_mask = outliers_mask[sort_indices]

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
            
            if treatment == 'winsorize':
                # Usar numpy para cálculos de percentis (mais rápido)
                p05, p95 = np.percentile(qty_values, [5, 95])
                if (p95 - p05) < 0.1 * mean_qty:
                    p05 = max(0, mean_qty - 2 * std_qty)
                    p95 = mean_qty + 2 * std_qty
                group_copy.loc[outliers_mask, 'qty'] = np.clip(group_copy.loc[outliers_mask, 'qty'], p05, p95)
            
            elif treatment == 'cap':
                if n_obs >= min_observations:
                    upper_limit = rolling_mean[outliers_mask] + 3 * rolling_std[outliers_mask]
                    lower_limit = np.maximum(rolling_mean[outliers_mask] - 3 * rolling_std[outliers_mask], 0)
                else:
                    upper_limit = mean_qty + 3 * std_qty
                    lower_limit = max(0, mean_qty - 3 * std_qty)
                group_copy.loc[outliers_mask, 'qty'] = np.clip(group_copy.loc[outliers_mask, 'qty'], lower_limit, upper_limit)

            elif treatment == 'remove':
                group_copy = group_copy[~outliers_mask]
            
            return group_copy, group_stats
        else:
            return group, group_stats

    except Exception as e:
        # Retornar o grupo original e estatísticas vazias em caso de erro
        return group, {}
        return group, {}

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
        
        print(f"   → Dados agregados iniciais: {len(self.df_agg):,} registros")
        
        # OPÇÃO 1: Não fazer cross join completo (mais eficiente)
        # Apenas garantir que cada combinação PDV-produto tenha pelo menos as semanas onde vendeu
        
        # OPÇÃO 2: Fazer cross join apenas para combinações que realmente venderam
        # Isto é muito mais inteligente que criar 53M registros desnecessários
        
        # Vamos usar OPÇÃO 1 por enquanto e ver se o modelo funciona bem
        # Se precisarmos de continuidade temporal, podemos implementar de forma mais eficiente
        
        print(f"   → Dados agregados finais: {len(self.df_agg):,} registros")
        print(f"   → PDVs únicos: {self.df_agg['pdv'].nunique():,}")
        print(f"   → Produtos únicos: {self.df_agg['internal_product_id'].nunique():,}")
        print(f"   → Range de semanas: {self.df_agg['week_of_year'].min()}-{self.df_agg['week_of_year'].max()}")
        
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
    
    def detect_and_treat_outliers(self, rolling_window=12, z_threshold=5.0, treatment='winsorize', 
                                  adaptive_threshold=True, min_observations=10):
        """
        Detecta e trata outliers usando rolling Z-score personalizado por PDV-produto
        
        Args:
            rolling_window (int): Tamanho da janela rolling para calcular média e desvio
            z_threshold (float): Threshold base do Z-score para considerar outlier
            treatment (str): Tipo de tratamento ('winsorize', 'remove', 'cap')
            adaptive_threshold (bool): Se True, ajusta threshold baseado nas características do grupo
            min_observations (int): Mínimo de observações para calcular estatísticas confiáveis
        """
        return self._detect_and_treat_outliers(rolling_window, z_threshold, treatment, 
                                                           adaptive_threshold, min_observations)

    def _detect_and_treat_outliers(self, rolling_window=12, z_threshold=5.0, 
                                               treatment='winsorize', adaptive_threshold=True, 
                                               min_observations=10):
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
            'min_observations': min_observations
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
        
        # Verificar se as colunas necessárias existem
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
    
    # Tratamento de outliers personalizado
    if treat_outliers:
        processor.detect_and_treat_outliers(**outlier_params)
        # Análise dos resultados
        outlier_analysis = processor.analyze_outlier_treatment()
    else:
        outlier_analysis = None
    
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
        'stats': stats,
        'outlier_analysis': outlier_analysis
    }