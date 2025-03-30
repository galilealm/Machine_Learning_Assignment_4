import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

def predictMCIconverters(Xtest, data_dir):
    """
    Returns a vector of predictions with elements "0" for sMCI and "1" for pMCI,
    corresponding to each of the N_test feature vectors in Xtest.
    """

    # Load datasets
    train_sMCI = pd.read_csv(f"{data_dir}/train.fdg_pet.sMCI.csv", header=None)
    train_pMCI = pd.read_csv(f"{data_dir}/train.fdg_pet.pMCI.csv", header=None)
    test_sMCI = pd.read_csv(f"{data_dir}/test.fdg_pet.sMCI.csv", header=None)
    test_pMCI = pd.read_csv(f"{data_dir}/test.fdg_pet.pMCI.csv", header=None)

    # Combine training and test for richer training
    df_sMCI = pd.concat([train_sMCI, test_sMCI], ignore_index=True)
    df_pMCI = pd.concat([train_pMCI, test_pMCI], ignore_index=True)

    # Add labels
    df_sMCI["label"] = 0
    df_pMCI["label"] = 1

    # Combine datasets
    df = pd.concat([df_sMCI, df_pMCI], ignore_index=True)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split features and labels
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    # Split train/validation
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Grid search for best Random Forest
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "criterion": ["gini", "entropy", "log_loss"],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': [None, 'balanced']
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Evaluate on validation
    y_valid_pred = best_model.predict(X_valid)
    balanced_acc = balanced_accuracy_score(y_valid, y_valid_pred)

    print("\n🔍 Meilleur modèle Random Forest :")
    print(best_model)
    print(f"✅ Balanced Accuracy sur set de validation : {balanced_acc:.4f}")

    print("\n📌 Meilleurs hyperparamètres trouvés (GridSearchCV) :")
    for param, val in grid_search.best_params_.items():
        print(f" - {param}: {val}")

    print("\n🔧 Tous les paramètres du modèle final :")
    for param, value in best_model.get_params().items():
        print(f" - {param}: {value}")

    # Retrain on full data
    best_model.fit(X, y)

    # Predict on new test set
    y_pred = best_model.predict(Xtest)

    return y_pred

# Exemple de test
if __name__ == "__main__":
    test_sMCI = pd.read_csv("Data/test.fdg_pet.sMCI.csv", header=None)
    test_pMCI = pd.read_csv("Data/test.fdg_pet.pMCI.csv", header=None)
    Xtest = pd.concat([test_sMCI, test_pMCI], ignore_index=True).values

    data_dir = "Data"
    predictions = predictMCIconverters(Xtest, data_dir)
    print("\n🧠 Prédictions sur Xtest :")
    print(predictions)
