import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import random

def calculate_separation_score(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    between_variance = 0
    within_variance = 0
    
    overall_mean = np.mean(X_scaled, axis=0)
    
    unique_targets = np.unique(y)
    n_total = len(y)
    
    for target in unique_targets:
        mask = (y == target)
        group_data = X_scaled[mask]
        group_mean = np.mean(group_data, axis=0)
        n_group = len(group_data)
        
        between_variance += n_group * np.sum((group_mean - overall_mean) ** 2)
        
        for point in group_data:
            within_variance += np.sum((point - group_mean) ** 2)
    
    if within_variance == 0:
        return float('inf')
    
    return between_variance / within_variance

def rkcnn_feature_selection(df, target_column, h=50, r=10, m_features=None, k=5, top_features=20):
    feature_columns = [col for col in df.columns if col not in [target_column, 'Age', 'SEX']]
    
    if m_features is None:
        m_features = max(1, len(feature_columns) // 3)
    
    X_features = df[feature_columns].values
    y = df[target_column].values
    
    separation_scores = []
    feature_sets = []
    
    for j in range(h):
        selected_features = random.sample(feature_columns, min(m_features, len(feature_columns)))
        feature_sets.append(selected_features)
        
        X_subset = df[selected_features].values
        score = calculate_separation_score(X_subset, y)
        separation_scores.append(score)
    
    sorted_indices = np.argsort(separation_scores)[::-1]
    sorted_scores = [separation_scores[i] for i in sorted_indices]
    sorted_feature_sets = [feature_sets[i] for i in sorted_indices]
    
    top_r = min(r, len(sorted_scores))
    weights = []
    models = []
    
    for j in range(top_r):
        weight = sorted_scores[j] / sum(sorted_scores[:top_r])
        weights.append(weight)
        
        X_subset = df[sorted_feature_sets[j]].values
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_subset, y)
        models.append((knn, sorted_feature_sets[j]))
    
    feature_importance = {}
    for feature in feature_columns:
        feature_importance[feature] = 0
    
    for j, (model, features) in enumerate(models):
        feature_weight = weights[j] / len(features)
        for feature in features:
            feature_importance[feature] += feature_weight
    
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_feature_names = [feature for feature, importance in sorted_features[:top_features]]
    
    result_columns = ['Age', 'SEX'] + top_feature_names
    result_df = df[result_columns].copy()
    
    return result_df

def process_csv(file_path, target_column='Age', h=50, r=10, top_features=20):
    df = pd.read_excel(file_path)

    if 'SEX' in df.columns:
        df['SEX'] = pd.Categorical(df['SEX']).codes

    result_df = rkcnn_feature_selection(df, target_column, h=h, r=r, top_features=top_features)
    
    return result_df

if __name__ == "__main__":
    file_path = "QC_removed_raw_sheet_valid_features.xlsx"
    selected_features_df = process_csv(file_path, target_column='Age', top_features=20)
    
    print(f"Selected {len(selected_features_df.columns)} features:")
    print(selected_features_df.columns.tolist())
    
    selected_features_df.to_csv("selected_features.csv", index=False)