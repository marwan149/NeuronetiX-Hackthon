from .data.load_data import load_data
from .feature_engineering import (
    add_domain_features,
    clean_raw_data,
    drop_uninformative_features,
    encode_categorical,
    scale_numeric,
    split_features_target,
)
from .modeling import (
    balance_with_smoteenn,
    build_classifier_grid,
    evaluate_classifier,
    run_grid_search,
    split_train_test,
)
from .config import DEFAULT_FILENAME


def build_model_dataset(filename: str = DEFAULT_FILENAME):
    df = load_data(filename)
    df = clean_raw_data(df)
    df = encode_categorical(df)
    df = add_domain_features(df)
    df = drop_uninformative_features(df)

    X, y = split_features_target(df)
    X = scale_numeric(X)
    return X, y, df


def run_classification_pipeline(filename: str = DEFAULT_FILENAME):
    X, y, df = build_model_dataset(filename)
    X_res, y_res = balance_with_smoteenn(X, y)
    X_train, X_test, y_train, y_test = split_train_test(X_res, y_res)

    results = {}
    for name, classifier_info in build_classifier_grid().items():
        print(f"Running {name}...")
        best_model, best_score = run_grid_search(
            classifier_info["model"],
            classifier_info["params"],
            X_train,
            y_train,
        )
        evaluation = evaluate_classifier(best_model, X_test, y_test)
        results[name] = {
            "best_estimator": best_model,
            "best_score": best_score,
            **evaluation,
        }

    return results


def main(filename: str = DEFAULT_FILENAME):
    results = run_classification_pipeline(filename)
    print("\nSummary")
    for name, info in results.items():
        print(f"{name}: ROC AUC {info['roc_auc']:.4f} (best search score {info['best_score']:.4f})")


if __name__ == "__main__":
    main()
