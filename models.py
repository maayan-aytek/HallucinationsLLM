import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from data_preparation import prepare_sentences_df
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from functools import partial
from utils import feature_to_target_corr, train_and_eval
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, precision_recall_fscore_support

# Load the dataset
df = pd.read_pickle("/home/student/HallucinationsLLM/data/sentences_df_all_features.pkl")

# Feature set and labels
FEATURES = ['sentence_image_similarity',
            'sentence_vec_dim_6',
            'max_pos_prob',
            'sentence_normalized_index',
            'sentence_vec_dim_1',
            'sentence_vec_dim_4',
            'sentence_vec_dim_7',
            'image_vec_dim_9',
            'sentence_vec_dim_2',
            'max_sentence_probes',
            'min_pos_prob',
            'sentence_vec_dim_9',
            'sentence_len',
            'sentence_vec_dim_8',
            'image_vec_dim_3',
            'median_sentence_probes',
            'objects_num',
            'median_sentence_entropy',
            'mean_pos_prob',
            'sentence_vec_dim_5',
            'sentence_vec_dim_10',
            'max_sentence_entropy',
            'sentence_vec_dim_3',
            'mean_r',
            'median_pos_prob',
            'mean_sentence_entropy',
            'image_vec_dim_4',
            'min_sentence_probes',
            'image_vec_dim_5',
            'mean_b',
            'image_vec_dim_6',
            'image_vec_dim_10',
            'mean_g',
            'image_vec_dim_7',
            'image_vec_dim_1',
            'mean_sentence_probes',
            'mean_rbg',
            'image_vec_dim_2',
            'min_sentence_entropy',
            'image_vec_dim_8',
            'sentence_contains_hedges',
            'contains_JJR',
            'contains_NNS',
            'contains_CD']

LABEL = 'sentences_labels'

X = df[FEATURES].astype(float)
y = df[LABEL].astype(int)

# FEATURES = ['contains_CD', 'max_pos_prob', 'sentence_len', 'min_pos_prob', 'max_sentence_entropy', 'min_sentence_entropy', 'max_sentence_probes', 
#             'mean_sentence_probes', 'min_sentence_probes', 'mean_pos_prob', 'median_sentence_probes', 'mean_sentence_entropy']
# FEATURES = ['sentence_normalized_index', 'sentence_contains_hedges', 'sentence_len', 'mean_pos_prob', 'max_pos_prob', 'min_pos_prob', 'median_pos_prob',
#             'median_sentence_probes', 'mean_sentence_probes', 'min_sentence_probes', 'max_sentence_probes', 'sentence_image_similarity'] #, 'objects_num']

# # feature_to_target_corr(df, FEATURES, LABEL, save_path="correlation_plot.png")

# xgb_model_class = partial(XGBClassifier, n_estimators=200, max_depth=50, eta=0.05, gamma=0.1, reg_lambda=0.8, min_child_weight=2, random_state=42)
# xgb_model, data_split, results = train_and_eval(df, xgb_model_class, FEATURES, LABEL)
# print(results)


def objective(trial, model_name="xgb"):
    # Suggest hyperparameters to tune
    if model_name == "xgb":
        param = {
            'max_depth': trial.suggest_int('max_depth', 10, 60),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.3, 0.8),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_uniform('gamma', 0, 10),
            'reg_alpha': trial.suggest_uniform('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_uniform('reg_lambda', 0, 10),
        }
        clf = XGBClassifier(**param)

    elif model_name == "rf":
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 10, 60),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }
        clf = RandomForestClassifier(**param)
    
    num_selected_features = trial.suggest_categorical('num_features', [10, 15, 20, 44])
    selected_features = FEATURES[:num_selected_features]
    X_selected = X[selected_features] 
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    score = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy').mean()
    return score

# Create Optuna study
model_name = 'rf'
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, model_name=model_name), n_trials=2000)

# Output the best parameters
print("Best hyperparameters:", study.best_params)
best_params = study.best_params
num_selected_features = best_params.pop('num_features')
selected_features = FEATURES[:num_selected_features]

if model_name == 'xgb':
    model_class = partial(XGBClassifier, **best_params)
elif model_name == 'rf':
    model_class = partial(RandomForestClassifier, **best_params)
    
model, data_split, results = train_and_eval(df, model_class, selected_features, LABEL)
print(results)
