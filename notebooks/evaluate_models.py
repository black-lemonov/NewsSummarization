import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import defaultdict
import os


def evaluate_models_on_datasets(models, datasets, save_path='./'):
    """
    Обучает несколько моделей на каждом датасете, вычисляет метрики, сохраняет графики, 
    таблицы и схемы деревьев (для DecisionTree)
    """
    plt.rcParams['font.family'] = 'DejaVu Sans'
    os.makedirs(save_path, exist_ok=True)

    if not isinstance(models, list):
        models = [models]
    
    results = defaultdict(lambda: {
        'num_features': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'roc_auc': [],
        'pr_auc': []  
    })

    for path in datasets:
        data = pd.read_csv(path, index_col=0)
        data = data.rename(
            columns={col: col.replace("(", "").replace(")", "").replace(",","").replace(".", "").replace(']', "") for col in data.columns}
        )
        X_train, X_test, y_train, y_test = train_test_split(
            data.drop(["labels", "text"], axis=1), 
            data["labels"], 
            test_size=0.2,
            random_state=42
        )
        del data
        
        n_features = X_train.shape[1]
        dataset_name = os.path.splitext(os.path.basename(path))[0] 
        
        for model in models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            model_name = getattr(model, 'name', model.__class__.__name__)
            results[model_name]['num_features'].append(n_features)
            results[model_name]['accuracy'].append(accuracy_score(y_test, y_pred))
            results[model_name]['precision'].append(precision_score(y_test, y_pred))
            results[model_name]['recall'].append(recall_score(y_test, y_pred))
            results[model_name]['roc_auc'].append(roc_auc_score(y_test, y_pred))
            results[model_name]['pr_auc'].append(average_precision_score(y_test, y_pred))

    for model_name in results:
        sorted_data = sorted(zip(
            results[model_name]['num_features'],
            results[model_name]['accuracy'],
            results[model_name]['precision'],
            results[model_name]['recall'],
            results[model_name]['roc_auc'],
            results[model_name]['pr_auc']
        ), key=lambda x: x[0])
        
        results[model_name] = {
            'num_features': [x[0] for x in sorted_data],
            'accuracy': [x[1] for x in sorted_data],
            'precision': [x[2] for x in sorted_data],
            'recall': [x[3] for x in sorted_data],
            'roc_auc': [x[4] for x in sorted_data],
            'pr_auc': [x[5] for x in sorted_data] 
        }

    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v']
    metric_info = {
        'accuracy': ('Достоверность (Accuracy)', 'accuracy_plot.png'),
        'precision': ('Точность (Precision)', 'precision_plot.png'),
        'recall': ('Полнота (Recall)', 'recall_plot.png'),
        'roc_auc': ('ROC-AUC', 'roc_auc_plot.png'),
        'pr_auc': ('PR-AUC', 'pr_auc_plot.png'),
    }
    
    for metric, (title, filename) in metric_info.items():
        plt.figure(figsize=(10, 6))
        
        for idx, (model_name, data) in enumerate(results.items()):
            plt.plot(
                data['num_features'], 
                data[metric],
                label=model_name,
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                linestyle='-',
                markersize=8
            )
        
        plt.title(title, fontsize=14)
        plt.xlabel('Количество признаков', fontsize=12)
        plt.ylabel(metric.split('_')[0].capitalize(), fontsize=12)
        plt.grid(True)
        plt.legend()
        
        plot_path = os.path.join(save_path, filename)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"График сохранен в {plot_path}!")
        
        if results:
            all_features = set()
            for data in results.values():
                all_features.update(data['num_features'])
            all_features = sorted(all_features)
            
            table_data = {}
            for model_name, data in results.items():
                metric_dict = dict(zip(data['num_features'], data[metric]))
                table_data[model_name] = [metric_dict.get(f, float('nan')) for f in all_features]
            
            df_table = pd.DataFrame(table_data, index=all_features)
            df_table.index.name = 'num_features'
            
            table_filename = filename.replace('_plot.png', '_data.csv')
            table_path = os.path.join(save_path, table_filename)
            df_table.to_csv(table_path, index=True)
            print(f"Таблица для метрики {metric} сохранена в {table_path}")