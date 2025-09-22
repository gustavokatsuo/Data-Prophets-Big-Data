#!/usr/bin/env python3
"""
Script de ajuda para mostrar como usar os diferentes modelos
"""

def show_help():
    """Mostra as opções disponíveis"""
    print("🚀 PIPELINE DE PREDIÇÃO DE VENDAS - GUIA DE USO")
    print("=" * 60)
    print()
    
    print("📋 OPÇÕES DISPONÍVEIS:")
    print("-" * 30)
    print()
    
    print("🤖 MODELOS DISPONÍVEIS:")
    print("  • LightGBM (padrão) - Rápido e eficiente")
    print("  • XGBoost - Robusto e versátil")
    print()
    
    print("💻 COMANDOS PARA EXECUÇÃO:")
    print("-" * 30)
    print()
    
    print("1️⃣ Pipeline completo com LightGBM (padrão):")
    print("   python main.py")
    print("   python main.py --model lightgbm")
    print()
    
    print("2️⃣ Pipeline completo com XGBoost:")
    print("   python main.py --model xgboost")
    print()
    
    print("3️⃣ Análise rápida dos dados (sem treinar modelo):")
    print("   python main.py --quick")
    print()
    
    print("4️⃣ Ver ajuda:")
    print("   python main.py --help")
    print()
    
    print("📦 INSTALAÇÃO DO XGBOOST:")
    print("-" * 30)
    print("Se você não tem XGBoost instalado, execute:")
    print("   python install_xgboost.py")
    print("Ou manualmente:")
    print("   pip install xgboost")
    print()
    
    print("📊 DIFERENÇAS ENTRE OS MODELOS:")
    print("-" * 30)
    print("LightGBM:")
    print("  ✅ Mais rápido para treinar")
    print("  ✅ Menor uso de memória")
    print("  ✅ Boa performance geral")
    print()
    print("XGBoost:")
    print("  ✅ Mais robusto com dados ruidosos")
    print("  ✅ Melhor para datasets complexos")
    print("  ✅ Mais opções de regularização")
    print()
    
    print("📁 OUTPUTS GERADOS:")
    print("-" * 30)
    print("• Modelos salvos em: models/")
    print("• Predições em: output/submission_[modelo]_jan2023_[timestamp].csv")
    print("• Visualizações em: output/")
    print("• Dados para análise: data_for_model_*.csv")
    print()
    
    print("🎯 FORMATO DAS PREDIÇÕES:")
    print("-" * 30)
    print("As predições são salvas em CSV com colunas:")
    print("• semana (1-5 para janeiro 2023)")
    print("• pdv (ID do ponto de venda)")
    print("• produto (ID do produto)")
    print("• quantidade (unidades previstas)")
    print()
    
    print("🔧 EXEMPLOS DE USO:")
    print("-" * 30)
    print("# Treinar com LightGBM e gerar predições")
    print("python main.py")
    print()
    print("# Treinar com XGBoost e gerar predições")
    print("python main.py --model xgboost")
    print()
    print("# Comparar ambos os modelos")
    print("python main.py --model lightgbm")
    print("python main.py --model xgboost")
    print()

if __name__ == "__main__":
    show_help()