import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

def _fit_one_vs_rest(X, Y, model_type="logreg", max_iter=200, random_state=42):
    models = {}
    if model_type == "logreg":
        for col in Y.columns:
            y = Y[col].values
            clf = LogisticRegression(
                solver="saga",
                penalty="l2",
                max_iter=max_iter,
                class_weight="balanced",
                random_state=random_state,
            )
            clf.fit(X, y)
            models[col] = clf
    elif model_type == "gbdt":
        for col in Y.columns:
            y = Y[col].values
            clf = GradientBoostingClassifier(random_state=random_state)
            clf.fit(X, y)
            models[col] = clf
    else:
        raise ValueError("model_type must be 'logreg' or 'gbdt'")
    return models

def _predict_proba(models, X):
    cols = sorted(models.keys())
    P = []
    for c in cols:
        m = models[c]
        if hasattr(m, "predict_proba"):
            p = m.predict_proba(X)[:,1]
        else:
            z = m.decision_function(X)
            p = 1/(1+np.exp(-z))
        P.append(p)
    P = np.vstack(P).T
    return pd.DataFrame(P, columns=cols, index=range(len(X)))

def train_number_models(X, Y_whites, Y_pb, model_type="logreg", max_iter=200, random_state=42):
    models_white = _fit_one_vs_rest(X, Y_whites, model_type=model_type, max_iter=max_iter, random_state=random_state)
    models_pb = _fit_one_vs_rest(X, Y_pb, model_type=model_type, max_iter=max_iter, random_state=random_state)
    return models_white, models_pb

def predict_number_probs(models_white, models_pb, X_future_row):
    Pw = _predict_proba(models_white, X_future_row)
    Pp = _predict_proba(models_pb, X_future_row)
    Pw = Pw / Pw.sum(axis=1).values[:,None]
    Pp = Pp / Pp.sum(axis=1).values[:,None]
    return Pw.iloc[0], Pp.iloc[0]

def compute_feature_importances(models: dict, feature_names, model_type: str):
    """
    Return a pandas Series of global feature importances.

    - For 'logreg': mean(|coef|) across the one-vs-rest classifiers.
    - For 'gbdt'  : mean(feature_importances_) across the one-vs-rest classifiers.
    """
    import numpy as np
    import pandas as pd
    fn = list(feature_names)
    M = len(models)
    if M == 0:
        return pd.Series(dtype=float)

    if model_type == "logreg":
        acc = np.zeros(len(fn), dtype=float)
        for mdl in models.values():
            coefs = getattr(mdl, "coef_", None)
            if coefs is None:
                continue
            v = np.abs(coefs[0])
            if len(v) != len(fn):
                v = v[:len(fn)]
            acc[:len(v)] += v
        imp = acc / max(1, M)
        return pd.Series(imp, index=fn).sort_values(ascending=False)

    elif model_type == "gbdt":
        acc = np.zeros(len(fn), dtype=float)
        for mdl in models.values():
            v = getattr(mdl, "feature_importances_", None)
            if v is None:
                continue
            if len(v) != len(fn):
                v = v[:len(fn)]
            acc[:len(v)] += v
        imp = acc / max(1, M)
        return pd.Series(imp, index=fn).sort_values(ascending=False)

    else:
        raise ValueError("model_type must be 'logreg' or 'gbdt'")
