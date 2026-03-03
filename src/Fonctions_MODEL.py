from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
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


def print_cv_results(models, X, y, cv=5):
    """
    Affiche les résultats de validation croisée pour un ensemble de modèles.

    La fonction évalue chaque modèle du dictionnaire fourni avec
    `evaluate_regression_model_cv`, puis affiche un résumé formaté
    des métriques moyennes sur les jeux d'entraînement et de test.

    Paramètres
    ----------
    models : dict
        Dictionnaire associant un nom de modèle à un estimateur.
        Exemple :
        {"Linear Regression": model_1, "Random Forest": model_2}
    X : pd.DataFrame
        Matrice des variables explicatives transmise à la fonction d'évaluation.
    y : pd.Series
        Vecteur cible transmis à la fonction d'évaluation.
    cv : int, default=5
        Nombre de folds utilisés pour la validation croisée.

    Retours
    -------
    None
        La fonction n'a pas de valeur de retour.
        Elle affiche simplement les résultats dans la console.

    Notes
    -----
    - Les métriques affichées sont : R2, MAPE, MAE et RMSE.
    - Cette fonction est pensée comme un utilitaire de comparaison rapide,
      notamment dans un notebook.
    """
    for name, model in models.items():
        metrics = evaluate_regression_model_cv(model, X, y, cv=cv)

        print(f"\n{name}")
        print(
            f"Train | R²: {metrics['Train R2']:.4f} | "
            f"MAPE: {metrics['Train MAPE (%)']:.2f}% | "
            f"MAE: {metrics['Train MAE']:.4f} | "
            f"RMSE: {metrics['Train RMSE']:.4f}"
        )
        print(
            f"Test  | R²: {metrics['Test R2']:.4f} | "
            f"MAPE: {metrics['Test MAPE (%)']:.2f}% | "
            f"MAE: {metrics['Test MAE']:.4f} | "
            f"RMSE: {metrics['Test RMSE']:.4f}"
        )
