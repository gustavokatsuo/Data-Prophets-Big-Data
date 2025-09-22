# Pipeline principal - orquestra todo o workflow
import os
import sys
import argparse
import pickle
import hashlib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Imports dos módulos criados
from scripts.config import *
from scripts.data_processing import process_data
from scripts.visualization import create_visualizations
from scripts.model import train_model, generate_predictions
from scripts.xgboost_model import train_xgboost_model, generate_xgboost_predictions
from multiprocessing import cpu_count

# Configuração do cache
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(data_path, use_advanced_features, use_interactions):
    """Gera chave única para cache baseada nos parâmetros de processamento"""
    # Incluir timestamp do arquivo de dados para invalidar cache se dados mudarem
    try:
        file_mtime = os.path.getmtime(data_path)
    except:
        file_mtime = 0
    
    cache_string = f"{data_path}_{file_mtime}_{use_advanced_features}_{use_interactions}"
    return hashlib.md5(cache_string.encode()).hexdigest()

def save_data_cache(data_dict, cache_key):
    """Salva dados processados no cache"""
    cache_path = os.path.join(CACHE_DIR, f'processed_data_{cache_key}.pkl')
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"💾 Cache dos dados salvo: {cache_path}")
        return True
    except Exception as e:
        print(f"⚠️ Erro ao salvar cache: {e}")
        return False

def load_data_cache(cache_key):
    """Carrega dados processados do cache"""
    cache_path = os.path.join(CACHE_DIR, f'processed_data_{cache_key}.pkl')
    try:
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                data_dict = pickle.load(f)
            print(f"📂 Cache dos dados carregado: {cache_path}")
            return data_dict
        return None
    except Exception as e:
        print(f"⚠️ Erro ao carregar cache: {e}")
        return None

def clean_old_cache(max_files=5):
    """Remove arquivos de cache antigos, mantendo apenas os mais recentes"""
    try:
        cache_files = [f for f in os.listdir(CACHE_DIR) if f.startswith('processed_data_') and f.endswith('.pkl')]
        
        if len(cache_files) > max_files:
            # Ordenar por data de modificação
            cache_files.sort(key=lambda x: os.path.getmtime(os.path.join(CACHE_DIR, x)))
            
            # Remover arquivos mais antigos
            for old_file in cache_files[:-max_files]:
                old_path = os.path.join(CACHE_DIR, old_file)
                os.remove(old_path)
                print(f"🗑️ Cache antigo removido: {old_file}")
    except Exception as e:
        print(f"⚠️ Erro ao limpar cache: {e}")

def main(model_type='lightgbm', use_advanced_features=True, use_interactions=True, use_cache=True):
    """Pipeline principal do projeto"""
    print("🚀 INICIANDO PIPELINE DE PREDIÇÃO DE VENDAS")
    print("=" * 60)
    print(f"⏱️  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🤖 Modelo selecionado: {model_type.upper()}")
    print(f"🔬 Features avançadas: {'SIM' if use_advanced_features else 'NÃO'}")
    print(f"🤝 Features de interação: {'SIM' if use_interactions else 'NÃO'}")
    print(f"� Cache de dados: {'HABILITADO' if use_cache else 'DESABILITADO'}")
    print(f"�💻 Usando {cpu_count()} cores para processamento paralelo")
    print()
    
    # ===== ETAPA 1: PROCESSAMENTO DE DADOS (COM CACHE) =====
    print("📊 ETAPA 1: PROCESSAMENTO DE DADOS")
    print("-" * 40)
    
    # Gerar chave do cache
    cache_key = get_cache_key(DATA_PATH, use_advanced_features, use_interactions)
    
    # Tentar carregar do cache primeiro
    data_dict = None
    if use_cache:
        print("🔍 Verificando cache de dados processados...")
        data_dict = load_data_cache(cache_key)
    
    # Se não encontrou no cache ou cache desabilitado, processar dados
    if data_dict is None:
        if use_cache:
            print("❌ Cache não encontrado. Processando dados...")
        else:
            print("🔄 Cache desabilitado. Processando dados...")
        
        try:
            data_dict = process_data(
                DATA_PATH, 
                use_advanced_features=use_advanced_features,
                use_interactions=use_interactions
            )
            print("✅ Dados processados com sucesso!")
            
            # Salvar no cache para próximas execuções
            if use_cache:
                save_data_cache(data_dict, cache_key)
                clean_old_cache()  # Limpar caches antigos
                
        except Exception as e:
            print(f"❌ Erro no processamento de dados: {e}")
            return None
    else:
        print("✅ Dados carregados do cache com sucesso!")
    
    print()
    
    # ===== ETAPA 2: ANÁLISE VISUAL E INSIGHTS =====
    print("📈 ETAPA 2: ANÁLISE VISUAL E INSIGHTS")
    print("-" * 40)
    
    try:
        viz_results = create_visualizations(data_dict, output_dir=OUTPUT_DIR)
        print("✅ Análises visuais geradas com sucesso!")
    except Exception as e:
        print(f"❌ Erro na geração de visualizações: {e}")
        viz_results = {}
    
    print()
    
    # ===== ETAPA 3: TREINAMENTO DO MODELO =====
    print(f"🤖 ETAPA 3: TREINAMENTO DO MODELO {model_type.upper()}")
    print("-" * 40)
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if model_type.lower() == 'xgboost':
            model_path = os.path.join(MODELS_DIR, f'xgboost_model_{timestamp}.pkl')
            model, wmape = train_xgboost_model(data_dict, save_model_path=model_path)
            print("✅ Modelo XGBoost treinado com sucesso!")
        else:  # lightgbm (padrão)
            model_path = os.path.join(MODELS_DIR, f'lightgbm_model_{timestamp}.pkl')
            model, wmape = train_model(data_dict, save_model_path=model_path)
            print("✅ Modelo LightGBM treinado com sucesso!")
            
        print(f"📊 WMAPE de validação: {wmape:.4f}")
    except Exception as e:
        print(f"❌ Erro no treinamento do modelo: {e}")
        return None
    
    print()
    
    # ===== ETAPA 4: ANÁLISE DO MODELO =====
    print("🎯 ETAPA 4: ANÁLISE DO MODELO")
    print("-" * 40)
    
    try:
        model_viz_results = create_visualizations(
            data_dict, 
            model=model, 
            output_dir=OUTPUT_DIR
        )
        
        # Salvar feature importance
        if 'feature_importance' in model_viz_results:
            importance_path = os.path.join(OUTPUT_DIR, 'feature_importance.csv')
            model_viz_results['feature_importance'].to_csv(importance_path, index=False)
            print(f"💾 Feature importance salva em: {importance_path}")
        
        print("✅ Análise do modelo concluída!")
    except Exception as e:
        print(f"❌ Erro na análise do modelo: {e}")
    
    print()
    
    # ===== ETAPA 5: GERAÇÃO DE PREDIÇÕES =====
    print("🔮 ETAPA 5: GERAÇÃO DE PREDIÇÕES")
    print("-" * 40)
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if model_type.lower() == 'xgboost':
            predictions_path = os.path.join(OUTPUT_DIR, f'submission_xgboost_jan2023_{timestamp}.csv')
            predictions_df = generate_xgboost_predictions(
                model, 
                data_dict, 
                weeks=PREDICTION_WEEKS,
                parallel=True,
                save_path=predictions_path
            )
            print("✅ Predições XGBoost geradas com sucesso!")
        else:  # lightgbm (padrão)
            predictions_path = os.path.join(OUTPUT_DIR, f'submission_lightgbm_jan2023_{timestamp}.csv')
            predictions_df = generate_predictions(
                model, 
                data_dict, 
                weeks=PREDICTION_WEEKS,
                parallel=True,
                save_path=predictions_path
            )
            print("✅ Predições LightGBM geradas com sucesso!")
            
    except Exception as e:
        print(f"❌ Erro na geração de predições: {e}")
        return None
    
    print()
    
    # ===== ETAPA 6: ANÁLISE DAS PREDIÇÕES =====
    print("📊 ETAPA 6: ANÁLISE DAS PREDIÇÕES")
    print("-" * 40)
    
    try:
        pred_viz_results = create_visualizations(
            data_dict,
            predictions_df=predictions_df,
            output_dir=OUTPUT_DIR
        )
        print("✅ Análise das predições concluída!")
    except Exception as e:
        print(f"❌ Erro na análise das predições: {e}")
    
    print()
    
    # ===== RELATÓRIO FINAL =====
    print("✅ PIPELINE CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
    print(f"⏱️  Finalizado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Resultados salvos em: {OUTPUT_DIR}/")
    print(f"🤖 Modelo salvo em: {MODELS_DIR}/")
    print()
    
    # Estatísticas finais
    print("📈 RESUMO DOS RESULTADOS:")
    print(f"• Modelo usado: {model_type.upper()}")
    print(f"• Total de predições: {len(predictions_df):,}")
    print(f"• PDVs únicos: {predictions_df['pdv'].nunique():,}")
    print(f"• Produtos únicos: {predictions_df['produto'].nunique():,}")
    print(f"• WMAPE do modelo: {wmape:.4f}")
    
    # Estatísticas por semana
    pred_stats = predictions_df.groupby('semana')['quantidade'].sum()
    print("\n🔮 PREDIÇÕES POR SEMANA:")
    for week in PREDICTION_WEEKS:
        if week in pred_stats.index:
            print(f"  Semana {week}: {pred_stats[week]:>10,} unidades")
    
    print()
    print("🎉 Processo finalizado! Verifique os arquivos de output para mais detalhes.")
    
    return {
        'data': data_dict,
        'model': model,
        'predictions': predictions_df,
        'wmape': wmape
    }

def run_quick_analysis():
    """Execução rápida apenas com análise de dados (sem modelo)"""
    print("🚀 ANÁLISE RÁPIDA DOS DADOS")
    print("=" * 40)
    
    # Processar dados
    data_dict = process_data(DATA_PATH)
    
    # Gerar visualizações
    create_visualizations(data_dict, output_dir=OUTPUT_DIR)
    
    print("✅ Análise rápida concluída!")
    return data_dict

if __name__ == "__main__":
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description='Pipeline de Predição de Vendas')
    parser.add_argument('--model', 
                       choices=['lightgbm', 'xgboost'], 
                       default='lightgbm',
                       help='Tipo de modelo a ser usado (lightgbm ou xgboost)')
    parser.add_argument('--quick', 
                       action='store_true',
                       help='Execução rápida apenas com análise de dados')
    parser.add_argument('--basic-features', 
                       action='store_true',
                       help='Usar apenas features básicas (não criar features avançadas)')
    parser.add_argument('--no-interactions', 
                       action='store_true',
                       help='Não criar features de interação')
    parser.add_argument('--no-cache', 
                       action='store_true',
                       help='Desabilitar cache de dados (sempre reprocessar)')
    parser.add_argument('--clear-cache', 
                       action='store_true',
                       help='Limpar todo o cache e sair')
    
    args = parser.parse_args()
    
    # Função para limpar cache
    if args.clear_cache:
        print("🗑️ LIMPANDO CACHE DE DADOS...")
        try:
            import shutil
            if os.path.exists(CACHE_DIR):
                shutil.rmtree(CACHE_DIR)
                os.makedirs(CACHE_DIR, exist_ok=True)
                print("✅ Cache limpo com sucesso!")
            else:
                print("ℹ️ Diretório de cache não existe.")
        except Exception as e:
            print(f"❌ Erro ao limpar cache: {e}")
        sys.exit(0)
    
    # Executar pipeline baseado nos argumentos
    if args.quick:
        run_quick_analysis()
    else:
        main(
            model_type=args.model,
            use_advanced_features=not args.basic_features,
            use_interactions=not args.no_interactions,
            use_cache=not args.no_cache
        )  