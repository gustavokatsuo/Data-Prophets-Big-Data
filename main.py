# Pipeline principal - orquestra todo o workflow
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Imports dos módulos criados
from scripts.config import *
from scripts.data_processing import process_data
from scripts.visualization import create_visualizations
from scripts.model import train_model, generate_predictions

def main():
    """Pipeline principal do projeto"""
    print("🚀 INICIANDO PIPELINE DE PREDIÇÃO DE VENDAS")
    print("=" * 60)
    print(f"⏱️  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💻 Usando {cpu_count()} cores para processamento paralelo")
    print()
    
    # ===== ETAPA 1: PROCESSAMENTO DE DADOS =====
    print("📊 ETAPA 1: PROCESSAMENTO DE DADOS")
    print("-" * 40)
    
    try:
        data_dict = process_data(DATA_PATH)
        print("✅ Dados processados com sucesso!")
    except Exception as e:
        print(f"❌ Erro no processamento de dados: {e}")
        return None
    
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
    print("🤖 ETAPA 3: TREINAMENTO DO MODELO")
    print("-" * 40)
    
    try:
        model_path = os.path.join(MODELS_DIR, f'lightgbm_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
        model, wmape = train_model(data_dict, save_model_path=model_path)
        print("✅ Modelo treinado com sucesso!")
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
        predictions_path = os.path.join(OUTPUT_DIR, 'submission_jan2023.csv')
        predictions_df = generate_predictions(
            model, 
            data_dict, 
            weeks=PREDICTION_WEEKS,
            parallel=True,
            save_path=predictions_path
        )
        print("✅ Predições geradas com sucesso!")
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
    print(f"• Total de predições: {len(predictions_df):,}")
    print(f"• PDVs únicos: {predictions_df['pdv'].nunique():,}")
    print(f"• Produtos únicos: {predictions_df['internal_product_id'].nunique():,}")
    print(f"• WMAPE do modelo: {wmape:.4f}")
    
    # Estatísticas por semana
    pred_stats = predictions_df.groupby('week')['predicted_qty'].sum()
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
    # Verificar argumentos da linha de comando
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        run_quick_analysis()
    else:
        main()  