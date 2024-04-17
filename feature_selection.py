import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

def load_data():
    X_train_scaled = pd.read_csv('data/X_train_scaled.csv')
    y_train = pd.read_csv('data/y_train.csv').squeeze()
    return X_train_scaled, y_train

def train_decision_tree(X_train, y_train):
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)
    return dt_classifier

def train_logistic_regression(X_train, y_train):
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    return lr_model

def get_feature_importances(dt_classifier, lr_model, X_train_scaled):
    dt_importances = pd.DataFrame({
        'feature': X_train_scaled.columns,
        'importance': dt_classifier.feature_importances_
    }).sort_values(by='importance', ascending=False)

    lr_importances = pd.DataFrame({
        'feature': X_train_scaled.columns,
        'importance': abs(lr_model.coef_[0])
    }).sort_values(by='importance', ascending=False)

    return dt_importances, lr_importances

def select_important_features(dt_importances, lr_importances, N):
    selected_features_dt = dt_importances.head(N)['feature'].tolist()
    selected_features_lr = lr_importances.head(N)['feature'].tolist()
    important_features = set(selected_features_dt + selected_features_lr)
    return important_features

def main():
    # Load and preprocess data
    X_train_scaled, y_train = load_data()

    # Train models
    dt_classifier = train_decision_tree(X_train_scaled, y_train)
    lr_model = train_logistic_regression(X_train_scaled, y_train)

    # Get feature importances
    dt_importances, lr_importances = get_feature_importances(dt_classifier, lr_model, X_train_scaled)

    # Select important features
    N = 10  # Number of top features to select
    important_features = select_important_features(dt_importances, lr_importances, N)

    # Print the selected important features
    print("Important features from Decision Tree and Logistic Regression:")
    print(pd.DataFrame(list(important_features), columns=["Feature Name"]))

    # Print feature importances
    print("\nDecision Tree Feature Importances:")
    print(dt_importances.head(N))
    print("\nLogistic Regression Feature Importances:")
    print(lr_importances.head(N))

if __name__ == '__main__':
    main()
