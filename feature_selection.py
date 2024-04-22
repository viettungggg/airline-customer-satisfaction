import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def load_data():
    # Load scaled features
    X_train_scaled = pd.read_csv('data/X_train_scaled.csv')
    X_test_scaled = pd.read_csv('data/X_test_scaled.csv')

    # Load and map target variable
    order_mapping = {'neutral or dissatisfied': 0, 'satisfied': 1}
    y_train = pd.read_csv('data/y_train.csv').squeeze().map(order_mapping)
    y_test = pd.read_csv('data/y_test.csv').squeeze().map(order_mapping)

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_decision_tree(X_train, y_train):
    params = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    dt = DecisionTreeClassifier(random_state=42)
    dt_cv = GridSearchCV(dt, params, cv=5)
    dt_cv.fit(X_train, y_train)
    print(f"Best parameters for Decision Tree: {dt_cv.best_params_}")
    return dt_cv.best_estimator_

def train_logistic_regression(X_train, y_train):
    params = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    }
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_cv = GridSearchCV(lr, params, cv=5)
    lr_cv.fit(X_train, y_train)
    print(f"Best parameters for Logistic Regression: {lr_cv.best_params_}")
    return lr_cv.best_estimator_

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        auprc = average_precision_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        results[name] = {'AUC': auc, 'AUPRC': auprc, 'F1': f1}
    return results


def plot_evaluation(results):
    labels = ['AUC', 'AUPRC', 'F1']
    dt_scores = [results['Decision Tree'][label] for label in labels]
    lr_scores = [results['Logistic Regression'][label] for label in labels]

    x = np.arange(len(labels))

    # Make the plot
    plt.figure(figsize=(8, 6))
    plt.bar(x - 0.2, dt_scores, 0.4, label='Decision Tree')
    plt.bar(x + 0.2, lr_scores, 0.4, label='Logistic Regression')

    plt.ylabel('Scores')
    plt.title('Model Evaluation Metrics')
    plt.xticks(x, labels)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.savefig('graphs/model_evaluation.png', bbox_inches='tight')
    plt.close()


def main():
    X_train_scaled, X_test_scaled, y_train, y_test = load_data()

    dt_classifier = train_decision_tree(X_train_scaled, y_train)
    lr_model = train_logistic_regression(X_train_scaled, y_train)

    models = {'Decision Tree': dt_classifier, 'Logistic Regression': lr_model}
    results = evaluate_models(models, X_test_scaled, y_test)
    plot_evaluation(results)


if __name__ == '__main__':
    main()
