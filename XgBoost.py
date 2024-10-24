import pandas as pd
import numpy as np
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import json
import matplotlib.pyplot as plt

data = pd.read_csv('datewisedata.csv')
data.drop(columns=['Date'], inplace=True)
X = data[['temperature', 'humidity', 'precip', 'wind' ]]
y = data['fire']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                           param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

model = grid_search.best_estimator_
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig('confusion_matrix.png')

plt.figure(figsize=(8, 6))
plot_importance(model, max_num_features=len(X.columns))
plt.title('Feature Importance')
plt.savefig('feature_importance.png')

feature_importances = model.feature_importances_
features = X.columns
for feature, importance in zip(features, feature_importances):
    print(f'{feature}: {importance:.4f}')

xgb_trees = model.get_booster().get_dump(with_stats=True)
forest_data = {
    "n_estimators": model.n_estimators,
    "trees": xgb_trees,
    "classes": model.classes_.tolist() if hasattr(model, "classes_") else [0, 1]
}

with open('xgboost_model_params.json', 'w') as json_file:
    json.dump(forest_data, json_file)

