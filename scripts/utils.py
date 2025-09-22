# Utilitários gerais para o projeto
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json

def asymmetric_logcosh_objective(y_true, y_pred):
    # Parâmetro de assimetria: penaliza mais quando y_true é 0
    # Valores maiores que 1.0 aumentam a penalidade.
    asymmetric_penalty = 1.5 
    
    # Erro residual
    residual = y_pred - y_true
    
    # Gradiente (primeira derivada) da função Log-Cosh
    grad = np.tanh(residual)
    
    # Hessiano (segunda derivada) da função Log-Cosh
    hess = 1.0 - grad**2
    
    # Aplicar a penalidade assimétrica
    # Aumenta o gradiente (e o ajuste) quando o valor real era 0
    grad[y_true == 0] *= asymmetric_penalty
    hess[y_true == 0] *= asymmetric_penalty # Ajusta o hessiano também
    
    return grad, hess

def ensure_directories_exist(directories):
    """Garante que os diretórios existam"""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Diretório garantido: {directory}")

def save_experiment_log(results, log_path):
    """Salva log do experimento"""
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'wmape': float(results.get('wmape', 0)),
        'total_predictions': len(results.get('predictions', [])),
        'unique_pdvs': results.get('predictions', pd.DataFrame()).get('pdv', pd.Series()).nunique() if 'predictions' in results else 0,
        'unique_products': results.get('predictions', pd.DataFrame()).get('produto', pd.Series()).nunique() if 'predictions' in results else 0,
    }
    
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"📝 Log do experimento salvo em: {log_path}")

def load_experiment_log(log_path):
    """Carrega log de experimento"""
    with open(log_path, 'r') as f:
        return json.load(f)

def compare_models(log_paths):
    """Compara diferentes experimentos"""
    print("📊 COMPARAÇÃO DE MODELOS")
    print("=" * 50)
    
    for i, log_path in enumerate(log_paths):
        if os.path.exists(log_path):
            log_data = load_experiment_log(log_path)
            print(f"Modelo {i+1}:")
            print(f"  • Timestamp: {log_data['timestamp']}")
            print(f"  • WMAPE: {log_data['wmape']:.4f}")
            print(f"  • Predições: {log_data['total_predictions']:,}")
            print()

def validate_data_quality(df):
    """Valida qualidade dos dados"""
    print("🔍 VALIDAÇÃO DA QUALIDADE DOS DADOS")
    print("-" * 40)
    
    issues = []
    
    # Verificar valores nulos
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        issues.append(f"Valores nulos encontrados: {null_counts.sum()}")
    
    # Verificar duplicatas
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"Registros duplicados: {duplicates}")
    
    # Verificar valores negativos em quantity
    if 'quantity' in df.columns:
        negative_qty = (df['quantity'] < 0).sum()
        if negative_qty > 0:
            issues.append(f"Quantidades negativas: {negative_qty}")
    
    # Verificar datas
    if 'transaction_date' in df.columns:
        try:
            pd.to_datetime(df['transaction_date'])
        except:
            issues.append("Problemas no formato de datas")
    
    if issues:
        print("⚠️  Problemas encontrados:")
        for issue in issues:
            print(f"  • {issue}")
    else:
        print("✅ Dados passaram na validação de qualidade!")
    
    return len(issues) == 0

def memory_usage_report(df, name="DataFrame"):
    """Relatório de uso de memória"""
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"💾 {name}: {memory_mb:.2f} MB de memória")
    return memory_mb

def performance_timer(func):
    """Decorator para medir tempo de execução"""
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"⏱️  {func.__name__} executado em {duration:.2f} segundos")
        return result
    return wrapper

def create_submission_summary(predictions_df, output_path):
    """Cria resumo da submission"""
    summary = {
        'total_predictions': len(predictions_df),
        'unique_pdvs': predictions_df['pdv'].nunique(),
        'unique_products': predictions_df['produto'].nunique(),
        'weeks_covered': sorted(predictions_df['semana'].unique()),
        'total_quantity_predicted': predictions_df['quantidade'].sum(),
        'avg_quantity_per_prediction': predictions_df['quantidade'].mean(),
        'predictions_by_week': predictions_df.groupby('semana')['quantidade'].sum().to_dict(),
        'top_10_pdvs_by_volume': predictions_df.groupby('pdv')['quantidade'].sum().nlargest(10).to_dict(),
        'creation_timestamp': datetime.now().isoformat()
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Resumo da submission salvo em: {output_path}")
    return summary