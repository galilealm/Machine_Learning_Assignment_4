import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix

FEATURE_NAMES = [
    "ctx-lh-inferiorparietal", "ctx-lh-inferiortemporal", "ctx-lh-isthmuscingulate", "ctx-lh-middletemporal",
    "ctx-lh-posteriorcingulate", "ctx-lh-precuneus", "ctx-rh-isthmuscingulate", "ctx-rh-posteriorcingulate",
    "ctx-rh-inferiorparietal", "ctx-rh-middletemporal", "ctx-rh-precuneus", "ctx-rh-inferiortemporal",
    "ctx-lh-entorhinal", "ctx-lh-supramarginal"
]

def load_data(data_dir):
    """Load and preprocess training and test datasets."""
    sMCI_train = pd.read_csv(f"{data_dir}/train.fdg_pet.sMCI.csv", names=FEATURE_NAMES)
    pMCI_train = pd.read_csv(f"{data_dir}/train.fdg_pet.pMCI.csv", names=FEATURE_NAMES)
    sMCI_test = pd.read_csv(f"{data_dir}/test.fdg_pet.sMCI.csv", names=FEATURE_NAMES)
    pMCI_test = pd.read_csv(f"{data_dir}/test.fdg_pet.pMCI.csv", names=FEATURE_NAMES)

    X_train = pd.concat([sMCI_train, pMCI_train], axis=0).values
    y_train = np.array([0] * len(sMCI_train) + [1] * len(pMCI_train))
    X_test = pd.concat([sMCI_test, pMCI_test], axis=0).values
    y_test = np.array([0] * len(sMCI_test) + [1] * len(pMCI_test))

    return X_train, y_train, X_test, y_test

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    spec = tn / (tn + fp)

    print(f"{model_name} Evaluation:")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall (Sensitivity):", rec)
    print("Specificity:", spec)
    print("Balanced Accuracy:", bal_acc)

    return acc, prec, rec

def train_decision_tree(X_train, y_train, X_test, y_test):
    param_grid = {'criterion': ['gini', 'entropy', 'log_loss']}
    clf = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
    clf.fit(X_train, y_train)

    print("Best Decision Tree Criterion:", clf.best_params_)

    best_tree = clf.best_estimator_
    evaluate_model(best_tree, X_test, y_test, "Decision Tree")

    plt.figure(figsize=(20, 10))
    plot_tree(best_tree, filled=True, feature_names=FEATURE_NAMES, class_names=['sMCI', 'pMCI'])
    plt.show()

    return best_tree

def train_random_forest(X_train, y_train, X_test, y_test):
    param_grid = {'criterion': ['gini', 'entropy', 'log_loss'], 'n_estimators': [100, 200]}
    clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
    clf.fit(X_train, y_train)

    print("Best Random Forest Parameters:", clf.best_params_)

    best_rf = clf.best_estimator_
    evaluate_model(best_rf, X_test, y_test, "Random Forest")

    return best_rf



if __name__ == "__main__":
    data_dir = "Data"
    X_train, y_train, X_test, y_test = load_data(data_dir)

    print("\nTraining Decision Tree...")
    train_decision_tree(X_train, y_train, X_test, y_test)

    print("\nTraining Random Forest...")
    train_random_forest(X_train, y_train, X_test, y_test)

    # print("\nGenerating predictions with the best model...")
    # predictions = predictMCIconverters(X_test, data_dir)
    # print("Predictions:", predictions)
