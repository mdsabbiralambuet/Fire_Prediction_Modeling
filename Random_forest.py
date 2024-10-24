import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import json
import matplotlib.pyplot as plt

data = pd.read_csv('datewisedata.csv')
data.drop(columns=['Date'], inplace=True)
X = data[['temperature', 'humidity', 'precip', 'wind' ,'monthly_fire']]
y = data['fire']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, scoring='accuracy')
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

def extract_tree(tree):
    tree_ = tree.tree_
    feature_names = [X.columns[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]

    def recurse(node):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[node]
            threshold = tree_.threshold[node]
            return {
                "feature": name,
                "threshold": threshold,
                "left": recurse(tree_.children_left[node]),
                "right": recurse(tree_.children_right[node])
            }
        else:
            return {
                "value": tree_.value[node].tolist()
            }

    return recurse(0)

forest_data = {
    "n_estimators": model.n_estimators,
    "trees": [extract_tree(estimator) for estimator in model.estimators_],
    "classes": model.classes_.tolist()
}

with open('model_params.json', 'w') as json_file:
    json.dump(forest_data, json_file)
