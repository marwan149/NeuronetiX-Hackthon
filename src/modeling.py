import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.combine import SMOTEENN
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from typing import Any, Dict, Tuple
import pandas as pd


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def balance_with_smoteenn(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    sampler = SMOTEENN(random_state=random_state)
    return sampler.fit_resample(X, y)


def build_classifier_grid() -> Dict[str, Dict[str, Any]]:
    return {
        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=1000),
            "params": {
                "max_depth": [3, 4, 5, 6],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        },
        "XGBoost": {
            "model": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
            "params": {
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5],
                "n_estimators": [100, 200],
            },
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=1000),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(random_state=1000),
            "params": {
                "learning_rate": [0.01, 0.1],
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
            },
        },
        "Logistic Regression": {
            "model": LogisticRegression(random_state=1000, max_iter=1000),
            "params": {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l2"],
                "solver": ["lbfgs"],
            },
        },
        "AdaBoost": {
            "model": AdaBoostClassifier(random_state=1000),
            "params": {
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.1, 1],
            },
        },
        "Extra Trees": {
            "model": ExtraTreesClassifier(random_state=1000),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
            },
        },
        "SGD Classifier": {
            "model": SGDClassifier(random_state=1000, max_iter=1000),
            "params": {
                "loss": ["hinge", "log_loss"],
                "penalty": ["l2", "l1"],
                "alpha": [0.0001, 0.001],
            },
        },
        "Support Vector Machine": {
            "model": SVC(probability=True, random_state=1000),
            "params": {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"],
            },
        },
        "K-Nearest Neighbors": {
            "model": KNeighborsClassifier(),
            "params": {
                "n_neighbors": [3, 5, 7],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
            },
        },
    }


def run_grid_search(
    model: Any,
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 3,
    scoring: str = "roc_auc",
) -> Tuple[Any, float]:
    search = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=RepeatedStratifiedKFold(n_splits=cv, n_repeats=1, random_state=42),
        scoring=scoring,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_score_


def evaluate_classifier(
    classifier: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    y_pred = classifier.predict(X_test)
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix ({type(classifier).__name__})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.show()

    try:
        RocCurveDisplay.from_estimator(classifier, X_test, y_test)
        plt.title(f"ROC Curve ({type(classifier).__name__})")
        plt.show()
    except Exception:
        pass

    print(report)
    return {"roc_auc": roc_auc}
