import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import json
import matplotlib.pyplot as plt

data = pd.read_csv('datewisedata.csv')
data.drop(columns=['Date'], inplace=True)
X = data[['temperature', 'humidity', 'precip', 'wind']]
y = data['fire']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.5, 1.0],
    'base_estimator__max_depth': [1, 2, 3]
}

base_estimator = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(AdaBoostClassifier(base_estimator=base_estimator, random_state=42), 
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

feature_importances = model.feature_importances_
features = X.columns
for feature, importance in zip(features, feature_importances):
    print(f'{feature}: {importance:.4f}')

plt.figure(figsize=(8, 6))
plt.barh(features, feature_importances, color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importance for AdaBoost Model')
plt.savefig('feature_importance.png')

adaboost_trees = []
for i, est in enumerate(model.estimators_):
    est_tree = est.tree_
    tree_structure = {
        "index": i,
        "n_node_samples": est_tree.n_node_samples.tolist(),
        "feature": est_tree.feature.tolist(),
        "threshold": est_tree.threshold.tolist(),
        "impurity": est_tree.impurity.tolist(),
        "value": est_tree.value.tolist(),
    }
    adaboost_trees.append(tree_structure)

forest_data = {
    "n_estimators": model.n_estimators,
    "trees": adaboost_trees,
    "classes": model.classes_.tolist()
}

with open('adaboost_model_params.json', 'w') as json_file:
    json.dump(forest_data, json_file)

