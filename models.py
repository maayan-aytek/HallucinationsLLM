import pandas as pd
import numpy as np
from data_preparation import prepare_sentences_df
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
from utils import feature_to_target_corr
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, precision_recall_fscore_support

# Load the dataset
sentences_df = prepare_sentences_df('/home/student/HallucinationsLLM/data/team5_clean_dataset.xlsx')

# Add new features
sentences_df['contains_CD'] = sentences_df['sentence_POS'].apply(lambda x: 'CD' in x).astype(int)
sentences_df['contains_NNS'] = sentences_df['sentence_POS'].apply(lambda x: 'NNS' in x).astype(int)
sentences_df['contains_JJR'] = sentences_df['sentence_POS'].apply(lambda x: 'JJR' in x).astype(int)
# Feature set and labels
FEATURES = ['sentence_normalized_index', 'sentence_contains_hedges', 'sentence_len', 'mean_pos_prob', 'max_pos_prob', 'min_pos_prob', 'median_pos_prob',
            'median_sentence_probes', 'mean_sentence_probes', 'min_sentence_probes', 'max_sentence_probes',
            # ?
            'median_sentence_entropy', 'mean_sentence_entropy', 'min_sentence_entropy', 'max_sentence_entropy',
            'contains_CD', 'contains_NNS', 'contains_JJR', 'objects_num', 'mean_b', 'mean_g', 'mean_r', 'mean_rbg']

# FEATURES = ['contains_CD', 'max_pos_prob', 'sentence_len', 'min_pos_prob', 'max_sentence_entropy', 'min_sentence_entropy', 'max_sentence_probes', 
#             'mean_sentence_probes', 'min_sentence_probes', 'mean_pos_prob', 'median_sentence_probes', 'mean_sentence_entropy']

FEATURES = ['sentence_normalized_index', 'sentence_contains_hedges', 'sentence_len', 'mean_pos_prob', 'max_pos_prob', 'min_pos_prob', 'median_pos_prob',
            'median_sentence_probes', 'mean_sentence_probes', 'min_sentence_probes', 'max_sentence_probes', 'sentence_image_similarity'] #, 'objects_num']

LABEL = 'sentences_labels'

feature_to_target_corr(sentences_df, FEATURES, LABEL, save_path="correlation_plot.png")

# Split the data into train and test sets
X = sentences_df[FEATURES].astype(float)
y = sentences_df[LABEL].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForest classifier
# rf = RandomForestClassifier(random_state=42, min_samples_leaf=3)
model = xgb.XGBClassifier(n_estimators=200, max_depth=50, eta=0.05, gamma=0.1, reg_lambda=0.8, min_child_weight=2)


# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC AUC score

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Feature Importance:")
print(feature_importance)

# Calculate classification metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
conf_matrix = confusion_matrix(y_test, y_pred)

# Print detailed classification report
report = classification_report(y_test, y_pred)
print("\nClassification Report:\n", report)

# Print individual metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nConfusion Matrix:\n", conf_matrix)
