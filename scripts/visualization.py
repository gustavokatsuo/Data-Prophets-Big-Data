# Módulo para análise visual e insights dos dados
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from .config import PLOT_CONFIG
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo dos gráficos
plt.style.use(PLOT_CONFIG['style'])
sns.set_palette(PLOT_CONFIG['palette'])

class DataVisualizer:
    """Classe para análises visuais e insights dos dados"""
    
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        
    def plot_data_insights(self, df_raw, df_agg, save_plots=True):
        """Gera insights visuais dos dados"""
        print("📊 Gerando insights visuais dos dados...")
        
        # Criar figura com subplots
        fig, axes = plt.subplots(2, 3, figsize=PLOT_CONFIG['figsize'])
        fig.suptitle('📊 Análise Exploratória dos Dados', fontsize=16, fontweight='bold')
        
        # 1. Distribuição de vendas ao longo do ano
        weekly_sales = df_agg.groupby('week_of_year')['qty'].sum()
        axes[0, 0].plot(weekly_sales.index, weekly_sales.values, linewidth=2, color='steelblue')
        axes[0, 0].set_title('📈 Vendas Totais por Semana')
        axes[0, 0].set_xlabel('Semana do Ano')
        axes[0, 0].set_ylabel('Quantidade Vendida')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Top 10 PDVs por volume
        top_pdvs = df_agg.groupby('pdv')['qty'].sum().nlargest(10)
        axes[0, 1].barh(range(len(top_pdvs)), top_pdvs.values, color='lightcoral')
        axes[0, 1].set_yticks(range(len(top_pdvs)))
        
        # Tratar PDVs que podem ser strings
        pdv_labels = []
        for x in top_pdvs.index:
            try:
                pdv_labels.append(f'PDV {int(x)}')
            except (ValueError, TypeError):
                pdv_labels.append(f'PDV {str(x)[:8]}')
        axes[0, 1].set_yticklabels(pdv_labels)
        axes[0, 1].set_title('🏪 Top 10 PDVs por Volume')
        axes[0, 1].set_xlabel('Quantidade Vendida')
        
        # 3. Distribuição de vendas (log scale)
        non_zero_sales = df_agg[df_agg['qty'] > 0]['qty']
        axes[0, 2].hist(np.log1p(non_zero_sales), bins=50, alpha=0.7, color='mediumseagreen')
        axes[0, 2].set_title('📊 Distribuição de Vendas (log+1)')
        axes[0, 2].set_xlabel('log(Quantidade + 1)')
        axes[0, 2].set_ylabel('Frequência')
        
        # 4. Correlação entre features
        feature_cols = ['qty', 'gross'] + [c for c in df_agg.columns if c.startswith('lag_')][:4]
        corr_matrix = df_agg[feature_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
        axes[1, 0].set_title('🔥 Correlação entre Features')
        
        # 5. Zero-inflated nature
        zero_pct = (df_agg['qty'] == 0).mean() * 100
        non_zero_pct = 100 - zero_pct
        axes[1, 1].pie([zero_pct, non_zero_pct], labels=['Vendas = 0', 'Vendas > 0'], 
                       autopct='%1.1f%%', colors=['lightgray', 'skyblue'])
        axes[1, 1].set_title(f'🎯 Distribuição Zero-Inflated\n({zero_pct:.1f}% zeros)')
        
        # 6. Sazonalidade temporal
        if 'mes' in df_raw.columns:
            monthly_sales = df_raw.groupby('mes')['quantity'].sum()
        else:
            # Fallback: usar trimestres baseados em semanas
            df_agg_temp = df_agg.copy()
            df_agg_temp['trimestre'] = ((df_agg_temp['week_of_year'] - 1) // 13) + 1
            monthly_sales = df_agg_temp.groupby('trimestre')['qty'].sum()
            monthly_sales.index = ['T1', 'T2', 'T3', 'T4']
        
        axes[1, 2].bar(range(len(monthly_sales)), monthly_sales.values, color='plum')
        axes[1, 2].set_title('📅 Sazonalidade Temporal')
        axes[1, 2].set_xlabel('Período')
        axes[1, 2].set_ylabel('Quantidade Vendida')
        axes[1, 2].set_xticks(range(len(monthly_sales)))
        axes[1, 2].set_xticklabels(monthly_sales.index)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{self.output_dir}/data_insights.png', dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_training_results(self, model, X_train, y_train, X_val, y_val, features, save_plots=True):
        """Plota resultados do treinamento e feature importance"""
        print("🎯 Analisando resultados do treinamento...")
        
        # Predições usando nossa classe wrapper
        if hasattr(model, 'model'):
            # É nossa classe LightGBMModel
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            lgb_model = model.model
            best_iteration = model.best_iteration
        else:
            # É o modelo do LightGBM diretamente
            try:
                train_pred = model.predict(X_train, num_iteration=model.best_iteration)
                val_pred = model.predict(X_val, num_iteration=model.best_iteration)
            except TypeError:
                train_pred = model.predict(X_train, iteration=model.best_iteration)
                val_pred = model.predict(X_val, iteration=model.best_iteration)
            lgb_model = model
            best_iteration = model.best_iteration
        
        # Criar figura
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🎯 Análise do Modelo LightGBM', fontsize=16, fontweight='bold')
        
        # 1. Feature Importance
        feature_imp = pd.DataFrame({
            'feature': features,
            'importance': lgb_model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False).head(15)
        
        axes[0, 0].barh(range(len(feature_imp)), feature_imp['importance'], color='steelblue')
        axes[0, 0].set_yticks(range(len(feature_imp)))
        axes[0, 0].set_yticklabels(feature_imp['feature'])
        axes[0, 0].set_title('🔥 Top 15 Features mais Importantes')
        axes[0, 0].set_xlabel('Importância (Gain)')
        
        # 2. Predições vs Real (Validação)
        sample_size = min(5000, len(val_pred))
        idx = np.random.choice(len(val_pred), sample_size, replace=False)
        axes[0, 1].scatter(y_val.iloc[idx], val_pred[idx], alpha=0.5, color='coral')
        axes[0, 1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', linewidth=2)
        axes[0, 1].set_xlabel('Valores Reais')
        axes[0, 1].set_ylabel('Predições')
        axes[0, 1].set_title('🎯 Predições vs Real (Validação)')
        
        # 3. Distribuição dos resíduos
        residuals = y_val - val_pred
        axes[1, 0].hist(residuals, bins=50, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('📊 Distribuição dos Resíduos')
        axes[1, 0].set_xlabel('Resíduos')
        axes[1, 0].set_ylabel('Frequência')
        axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
        
        # 4. Learning Curve
        if hasattr(model, 'evals_result_'):
            eval_results = model.evals_result_
            train_scores = eval_results['training']['l1']
            val_scores = eval_results['valid_1']['l1']
            
            axes[1, 1].plot(train_scores, label='Treino', color='blue', linewidth=2)
            axes[1, 1].plot(val_scores, label='Validação', color='orange', linewidth=2)
            axes[1, 1].set_title('📈 Curva de Aprendizado')
            axes[1, 1].set_xlabel('Iterações')
            axes[1, 1].set_ylabel('MAE')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{self.output_dir}/training_results.png', dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.show()
        
        return feature_imp, fig
    
    def plot_predictions_summary(self, predictions_df, save_plots=True):
        """Gera gráfico resumo das predições"""
        print("📊 Gerando resumo visual das predições...")
        
        plt.figure(figsize=(12, 6))
        week_totals = predictions_df.groupby('semana')['quantidade'].sum()
        plt.bar(week_totals.index, week_totals.values, color='steelblue', alpha=0.8)
        plt.title('🔮 Predições Totais por Semana - Janeiro 2023', fontsize=14, fontweight='bold')
        plt.xlabel('Semana')
        plt.ylabel('Quantidade Predita')
        plt.grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for i, v in enumerate(week_totals.values):
            plt.text(i+1, v + v*0.01, f'{v:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{self.output_dir}/predictions_summary.png', dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.show()
        
        return plt.gcf()
    
    def print_data_stats(self, stats):
        """Imprime estatísticas descritivas dos dados"""
        print("\n📈 ESTATÍSTICAS DESCRITIVAS:")
        print(f"• Total de registros: {stats['total_records']:,}")
        print(f"• PDVs únicos: {stats['unique_pdvs']:,}")
        print(f"• Produtos únicos: {stats['unique_products']:,}")
        print(f"• Vendas médias por semana: {stats['mean_sales']:.2f}")
        print(f"• Vendas mediana: {stats['median_sales']:.2f}")
        print(f"• Percentual de zeros: {stats['zero_percentage']:.1f}%")
        print(f"• Valor bruto médio: R$ {stats['mean_gross_value']:.2f}")
        print(f"• Range de semanas: {stats['weeks_range'][0]} - {stats['weeks_range'][1]}")
    
    def print_model_metrics(self, train_pred, val_pred, y_train, y_val):
        """Imprime métricas detalhadas do modelo"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        
        print(f"\n🎯 MÉTRICAS DETALHADAS:")
        print(f"• MAE Treino: {train_mae:.4f} | Validação: {val_mae:.4f}")
        print(f"• RMSE Treino: {train_rmse:.4f} | Validação: {val_rmse:.4f}")
        print(f"• R² Treino: {train_r2:.4f} | Validação: {val_r2:.4f}")
        print(f"• WMAPE Treino: {np.sum(np.abs(train_pred - y_train)) / np.sum(y_train):.4f}")
        print(f"• WMAPE Validação: {np.sum(np.abs(val_pred - y_val)) / np.sum(y_val):.4f}")
    
    def print_feature_importance(self, feature_importance):
        """Imprime top features mais importantes"""
        print(f"\n🔥 TOP 10 FEATURES MAIS IMPORTANTES:")
        for i, (feat, imp) in enumerate(feature_importance.head(10).values):
            print(f"{i+1:2d}. {feat:<15} → {imp:>8.0f}")
    
    def print_predictions_stats(self, predictions_df):
        """Imprime estatísticas das predições"""
        pred_stats = predictions_df.groupby('semana')['quantidade'].agg(['sum', 'mean', 'std'])
        print(f"\n📊 ESTATÍSTICAS DAS PREDIÇÕES POR SEMANA:")
        for week in [1, 2, 3, 4, 5]:
            if week in pred_stats.index:
                stats = pred_stats.loc[week]
                print(f"  Semana {week}: Total={stats['sum']:>8,.0f} | Média={stats['mean']:>6.1f} | Std={stats['std']:>6.1f}")

def create_visualizations(data_dict, model=None, predictions_df=None, output_dir='output'):
    """Função principal para criar todas as visualizações"""
    visualizer = DataVisualizer(output_dir)
    
    # Insights dos dados
    visualizer.plot_data_insights(data_dict['raw_data'], data_dict['aggregated_data'])
    visualizer.print_data_stats(data_dict['stats'])
    
    results = {}
    
    # Análise do modelo se fornecido
    if model is not None:
        X_train, y_train = data_dict['train_data']
        X_val, y_val = data_dict['validation_data']
        
        feature_importance, _ = visualizer.plot_training_results(
            model, X_train, y_train, X_val, y_val, data_dict['feature_columns']
        )
        
        # Predições para métricas
        if hasattr(model, 'model'):
            # É nossa classe LightGBMModel
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
        else:
            # É o modelo do LightGBM diretamente
            try:
                train_pred = model.predict(X_train, num_iteration=model.best_iteration)
                val_pred = model.predict(X_val, num_iteration=model.best_iteration)
            except TypeError:
                train_pred = model.predict(X_train, iteration=model.best_iteration)
                val_pred = model.predict(X_val, iteration=model.best_iteration)
        
        visualizer.print_model_metrics(train_pred, val_pred, y_train, y_val)
        visualizer.print_feature_importance(feature_importance)
        
        results['feature_importance'] = feature_importance
    
    # Análise das predições se fornecidas
    if predictions_df is not None:
        visualizer.plot_predictions_summary(predictions_df)
        visualizer.print_predictions_stats(predictions_df)
        results['predictions_plot'] = True
    
    return results