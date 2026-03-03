from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from category_encoders import BinaryEncoder


def encode_cat_col(
    df: pd.DataFrame,
    col_name: str,
    encoding_type: str,
) -> tuple[pd.DataFrame, object]:
    """
    Encode une seule variable catégorielle selon la méthode choisie.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame source.
    col_name : str
        Nom de la colonne à encoder.
    encoding_type : str
        Type d'encodage à appliquer.
        Valeurs possibles :
        - "onehot"
        - "binary"
        - "ordinal"

    Retours
    -------
    tuple[pd.DataFrame, object]
        - Le DataFrame avec la colonne encodée
        - L'encodeur entraîné

    Notes
    -----
    - `onehot` convient aux variables nominales à faible cardinalité.
    - `binary` convient aux variables nominales à forte cardinalité.
    - `ordinal` convient aux variables ordinales.
    - Pour `binary`, il faut installer `category-encoders`.
    """
    if col_name not in df.columns:
        raise ValueError(f"La colonne '{col_name}' n'existe pas dans le DataFrame.")

    df_encoded = df.copy()

    if encoding_type == "onehot":
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )

        encoded_array = encoder.fit_transform(df_encoded[[col_name]])
        encoded_cols = encoder.get_feature_names_out([col_name])

        encoded_df = pd.DataFrame(
            encoded_array,
            columns=encoded_cols,
            index=df_encoded.index,
        )

        df_encoded = pd.concat(
            [df_encoded.drop(columns=[col_name]), encoded_df],
            axis=1,
        )

        return df_encoded, encoder

    if encoding_type == "binary":
        encoder = BinaryEncoder(cols=[col_name])

        encoded_df = encoder.fit_transform(df_encoded[[col_name]])

        df_encoded = pd.concat(
            [df_encoded.drop(columns=[col_name]), encoded_df],
            axis=1,
        )

        return df_encoded, encoder

    if encoding_type == "ordinal":
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

        encoded_array = encoder.fit_transform(df_encoded[[col_name]])
        df_encoded[col_name] = encoded_array.astype(int)

        return df_encoded, encoder

    raise ValueError(
        "encoding_type doit être parmi : 'onehot', 'binary', 'ordinal'."
    )


def evaluate_regression_model(model, X, y, test_size=0.2):
    """
    Évalue un modèle de régression à l'aide d'un unique découpage train/test.

    La fonction sépare les données en un jeu d'entraînement et un jeu de test,
    entraîne le modèle sur le jeu d'entraînement, puis calcule plusieurs
    métriques de régression sur les deux sous-ensembles.

    Paramètres
    ----------
    model : estimator object
        Modèle de régression implémentant les méthodes `fit(X, y)` et `predict(X)`.
    X : pd.DataFrame ou array-like
        Matrice des variables explicatives.
    y : pd.Series ou array-like
        Vecteur cible.
    test_size : float, default=0.2
        Proportion des données réservée au jeu de test.

    Retours
    -------
    dict
        Dictionnaire contenant les métriques d'entraînement et de test :
        - "Train R2"
        - "Test R2"
        - "Train MAPE (%)"
        - "Test MAPE (%)"
        - "Train MAE"
        - "Test MAE"
        - "Train RMSE"
        - "Test RMSE"

    Notes
    -----
    - Le découpage est reproductible grâce à `random_state=42`.
    - Le `MAPE` est renvoyé en pourcentage.
    - Le `MAPE` peut devenir instable si la cible contient des valeurs proches de zéro.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        "Train R2": r2_score(y_train, y_train_pred),
        "Test R2": r2_score(y_test, y_test_pred),

        "Train MAPE (%)": mean_absolute_percentage_error(y_train, y_train_pred) * 100,
        "Test MAPE (%)": mean_absolute_percentage_error(y_test, y_test_pred) * 100,

        "Train MAE": mean_absolute_error(y_train, y_train_pred),
        "Test MAE": mean_absolute_error(y_test, y_test_pred),

        "Train RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
    }

    return metrics


def evaluate_regression_model_cv(model, X, y, cv):
    """
    Évalue un modèle de régression à l'aide d'une validation croisée K-Fold.

    La fonction réalise une validation croisée mélangée, entraîne le modèle
    sur chaque fold d'entraînement, évalue les performances sur les folds
    d'entraînement et de validation, puis renvoie la moyenne des métriques
    sur l'ensemble des folds.

    Paramètres
    ----------
    model : estimator object
        Modèle de régression implémentant les méthodes `fit(X, y)` et `predict(X)`.
    X : pd.DataFrame
        Matrice des variables explicatives. La fonction utilise `.iloc`,
        un DataFrame pandas est donc attendu.
    y : pd.Series
        Vecteur cible. La fonction utilise `.iloc`,
        une Series pandas est donc attendue.
    cv : int
        Nombre de folds à utiliser pour la validation croisée.

    Retours
    -------
    dict
        Dictionnaire contenant la moyenne des métriques d'entraînement
        et de test sur l'ensemble des folds :
        - "Train R2"
        - "Test R2"
        - "Train MAPE (%)"
        - "Test MAPE (%)"
        - "Train MAE"
        - "Test MAE"
        - "Train RMSE"
        - "Test RMSE"

    Notes
    -----
    - La validation croisée est reproductible grâce à `random_state=42`.
    - Le `MAPE` est renvoyé en pourcentage.
    - Le `MAPE` peut être difficile à interpréter si la cible contient
      des valeurs très faibles.
    - Les métriques d'entraînement sont elles aussi moyennées sur l'ensemble des folds.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    r2_train, r2_test = [], []
    mape_train, mape_test = [], []
    mae_train, mae_test = [], []
    rmse_train, rmse_test = [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)

        y_train_pred_real = model.predict(X_train)
        y_test_pred_real = model.predict(X_test)

        r2_train.append(r2_score(y_train, y_train_pred_real))
        r2_test.append(r2_score(y_test, y_test_pred_real))

        mape_train.append(mean_absolute_percentage_error(y_train, y_train_pred_real) * 100)
        mape_test.append(mean_absolute_percentage_error(y_test, y_test_pred_real) * 100)

        mae_train.append(mean_absolute_error(y_train, y_train_pred_real))
        mae_test.append(mean_absolute_error(y_test, y_test_pred_real))

        rmse_train.append(np.sqrt(mean_squared_error(y_train, y_train_pred_real)))
        rmse_test.append(np.sqrt(mean_squared_error(y_test, y_test_pred_real)))

    metrics = {
        "Train R2": np.mean(r2_train),
        "Test R2": np.mean(r2_test),

        "Train MAPE (%)": np.mean(mape_train),
        "Test MAPE (%)": np.mean(mape_test),

        "Train MAE": np.mean(mae_train),
        "Test MAE": np.mean(mae_test),

        "Train RMSE": np.mean(rmse_train),
        "Test RMSE": np.mean(rmse_test),
    }

    return metrics


def evaluate_classification_model_cv(model,X,y,cv,stratify=True,model_name=None,show_confusion_matrix=True,cmap="Blues",):
    """
    Évalue un modèle de classification à l'aide d'une validation croisée stratifiée.

    La fonction réalise une validation croisée `StratifiedKFold`, entraîne le
    modèle sur chaque fold d'entraînement, puis calcule plusieurs métriques de
    classification sur les jeux d'entraînement et de validation. Elle agrège
    également les prédictions de validation (out-of-fold) afin de produire une
    matrice de confusion représentative sur l'ensemble du jeu fourni, sans
    utiliser un jeu de test final mis de côté.

    Paramètres
    ----------
    model : estimator object
        Modèle de classification implémentant les méthodes `fit(X, y)` et
        `predict(X)`. Si le modèle expose `predict_proba(X)` ou
        `decision_function(X)`, un score ROC AUC est également calculé.
    X : pd.DataFrame
        Matrice des variables explicatives. La fonction utilise `.iloc`,
        un DataFrame pandas est donc attendu.
    y : pd.Series
        Vecteur cible. La fonction utilise `.iloc`,
        une Series pandas est donc attendue.
    cv : int
        Nombre de folds utilisés pour la validation croisée.
    stratify : bool, default=True
        Si `True`, utilise `StratifiedKFold` pour préserver la distribution
        des classes dans chaque fold. Si `False`, utilise `KFold`.
    model_name : str | None, default=None
        Nom du modèle à afficher dans la sortie console. Si `None`, seul le
        résumé des métriques est affiché.
    show_confusion_matrix : bool, default=True
        Si `True`, affiche la matrice de confusion agrégée à partir des
        prédictions out-of-fold.
    cmap : str, default="Blues"
        Palette de couleurs utilisée pour la matrice de confusion.

    Retours
    -------
    dict
        Dictionnaire contenant la moyenne des métriques d'entraînement et de
        validation sur l'ensemble des folds :
        - "Train Accuracy"
        - "Test Accuracy"
        - "Train Precision"
        - "Test Precision"
        - "Train Recall"
        - "Test Recall"
        - "Train F1"
        - "Test F1"
        - "Train ROC AUC"
        - "Test ROC AUC"

    Notes
    -----
    - Si `stratify=True`, la validation préserve la distribution des classes
      dans chaque fold.
    - Les métriques `Precision`, `Recall` et `F1` sont calculées avec`zero_division=0` pour éviter les erreurs en cas de classe non prédite.
    - Le score `ROC AUC` est calculé uniquement si le modèle fournit un score continu (`predict_proba` ou `decision_function`). Sinon, la métrique est renvoyée à `np.nan`.
    - La matrice de confusion est calculée sur les prédictions de validation agrégées sur l'ensemble des folds (out-of-fold).
    """
    splitter = (
        StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        if stratify
        else KFold(n_splits=cv, shuffle=True, random_state=42)
    )

    accuracy_train, accuracy_test = [], []
    precision_train, precision_test = [], []
    recall_train, recall_test = [], []
    f1_train, f1_test = [], []
    roc_auc_train, roc_auc_test = [], []
    y_true_oof = pd.Series(index=y.index, dtype=y.dtype)
    y_pred_oof = pd.Series(index=y.index, dtype=y.dtype)

    split_iterator = splitter.split(X, y) if stratify else splitter.split(X)

    for train_idx, test_idx in split_iterator:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_true_oof.iloc[test_idx] = y_test.to_numpy()
        y_pred_oof.iloc[test_idx] = y_test_pred

        accuracy_train.append(accuracy_score(y_train, y_train_pred))
        accuracy_test.append(accuracy_score(y_test, y_test_pred))

        precision_train.append(precision_score(y_train, y_train_pred, zero_division=0))
        precision_test.append(precision_score(y_test, y_test_pred, zero_division=0))

        recall_train.append(recall_score(y_train, y_train_pred, zero_division=0))
        recall_test.append(recall_score(y_test, y_test_pred, zero_division=0))

        f1_train.append(f1_score(y_train, y_train_pred, zero_division=0))
        f1_test.append(f1_score(y_test, y_test_pred, zero_division=0))

        train_scores = None
        test_scores = None

        if hasattr(model, "predict_proba"):
            train_scores = model.predict_proba(X_train)[:, 1]
            test_scores = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            train_scores = model.decision_function(X_train)
            test_scores = model.decision_function(X_test)

        if train_scores is not None and test_scores is not None:
            roc_auc_train.append(roc_auc_score(y_train, train_scores))
            roc_auc_test.append(roc_auc_score(y_test, test_scores))
        else:
            roc_auc_train.append(np.nan)
            roc_auc_test.append(np.nan)

    def _format_mean(values):
        """
        Calcule une moyenne et renvoie un float Python arrondi.

        Si toutes les valeurs sont manquantes, la fonction renvoie `nan`.
        """
        valid_values = [value for value in values if not np.isnan(value)]
        if not valid_values:
            return float("nan")
        return round(float(np.mean(valid_values)), 4)

    metrics = {
        "Train Accuracy": _format_mean(accuracy_train),
        "Test Accuracy": _format_mean(accuracy_test),

        "Train Precision": _format_mean(precision_train),
        "Test Precision": _format_mean(precision_test),

        "Train Recall": _format_mean(recall_train),
        "Test Recall": _format_mean(recall_test),

        "Train F1": _format_mean(f1_train),
        "Test F1": _format_mean(f1_test),

        "Train ROC AUC": _format_mean(roc_auc_train),
        "Test ROC AUC": _format_mean(roc_auc_test),
    }

    if model_name is not None:
        print(f"\n{model_name}")

    print(
        f"Train | Accuracy: {metrics['Train Accuracy']:.4f} | "
        f"Precision: {metrics['Train Precision']:.4f} | "
        f"Recall: {metrics['Train Recall']:.4f} | "
        f"F1: {metrics['Train F1']:.4f} | "
        f"ROC AUC: {metrics['Train ROC AUC']:.4f}"
    )
    print(
        f"Test  | Accuracy: {metrics['Test Accuracy']:.4f} | "
        f"Precision: {metrics['Test Precision']:.4f} | "
        f"Recall: {metrics['Test Recall']:.4f} | "
        f"F1: {metrics['Test F1']:.4f} | "
        f"ROC AUC: {metrics['Test ROC AUC']:.4f}"
    )

    if show_confusion_matrix:
        cm = confusion_matrix(y_true_oof, y_pred_oof)

        if cm.shape == (2, 2):
            labels = np.array(
                [
                    [f"Vrai negatif\n{cm[0, 0]}", f"Faux positif\n{cm[0, 1]}"],
                    [f"Faux negatif\n{cm[1, 0]}", f"Vrai positif\n{cm[1, 1]}"],
                ]
            )

            plt.figure(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=labels,
                fmt="",
                cmap=cmap,
                cbar=False,
                xticklabels=["Prediction negative", "Prediction positive"],
                yticklabels=["Reel negatif", "Reel positif"],
            )
            title = "Matrice de confusion agrégée (validation croisée)"
            if model_name is not None:
                title = f"{title} - {model_name}"
            plt.title(title)
            plt.xlabel("Prediction")
            plt.ylabel("Reel")
            plt.tight_layout()
            plt.show()
        else:
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=np.unique(y_true_oof),
            )
            disp.plot(cmap=cmap, values_format="d")
            title = "Matrice de confusion agrégée (validation croisée)"
            if model_name is not None:
                title = f"{title} - {model_name}"
            plt.title(title)
            plt.grid(False)
            plt.show()

    return metrics
