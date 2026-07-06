"""Tests for leak-free per-fold CV preprocessing.

FoldPreprocessor must learn imputation/scaling statistics from a fold's training
rows only and apply them unchanged to the validation rows. CVTrainer must invoke
it per fold so no out-of-fold statistics reach any fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fairxai.data.preprocessors import FoldPreprocessor, apply_fold_preprocessing
from fairxai.models.cv_trainer import CVTrainer


class TestFoldPreprocessor:
    def test_transform_scales_val_with_train_statistics(self) -> None:
        # Train [10, 20, 30]: mean=20, std(ddof=0)=~8.165. The validation rows
        # must be standardized with THOSE stats, never their own.
        train = pd.DataFrame({"bp": [10.0, 20.0, 30.0]})
        val = pd.DataFrame({"bp": [20.0, 30.0]})
        fp = FoldPreprocessor()

        fp.fit_transform(train)
        out = fp.transform(val)

        std = np.std([10.0, 20.0, 30.0])  # ddof=0, matches StandardScaler
        assert out["bp"].iloc[0] == 0.0  # (20 - 20) / std
        np.testing.assert_allclose(out["bp"].iloc[1], (30.0 - 20.0) / std)

    def test_missing_filled_with_train_median_not_val(self) -> None:
        # Train median of present [10, 20, 30] is 20. The validation NaN must be
        # filled with 20 (train median), then scaled with train stats -> 0.0.
        train = pd.DataFrame({"bp": [10.0, 20.0, 30.0]})
        val = pd.DataFrame({"bp": [np.nan]})
        fp = FoldPreprocessor()

        fp.fit_transform(train)
        out = fp.transform(val)

        assert not out["bp"].isnull().any()
        assert out["bp"].iloc[0] == 0.0  # (train_median 20 - train_mean 20) / std

    def test_all_missing_train_column_falls_back_without_nan(self) -> None:
        train = pd.DataFrame({"bp": [np.nan, np.nan]})
        val = pd.DataFrame({"bp": [np.nan]})
        fp = FoldPreprocessor()

        out_train = fp.fit_transform(train)
        out_val = fp.transform(val)

        assert not out_train["bp"].isnull().any()
        assert not out_val["bp"].isnull().any()

    def test_index_and_columns_preserved(self) -> None:
        val = pd.DataFrame({"bp": [1.0, 2.0], "hr": [3.0, 4.0]}, index=[7, 9])
        fp = FoldPreprocessor()
        fp.fit_transform(pd.DataFrame({"bp": [1.0, 2.0, 3.0], "hr": [3.0, 4.0, 5.0]}))

        out = fp.transform(val)
        assert list(out.index) == [7, 9]
        assert list(out.columns) == ["bp", "hr"]


class _RecordingModel:
    """Minimal model that records the mean of each fold's training matrix."""

    train_means: list = []

    def __init__(self, **_kwargs) -> None:
        self.training_metrics = {}

    def train(self, X, y):
        _RecordingModel.train_means.append(float(np.asarray(X, dtype=float).mean()))
        self.training_metrics = self._metrics()
        return self.training_metrics

    def evaluate(self, X, y):
        return self._metrics()

    @staticmethod
    def _metrics():
        return {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0,
            "auc_roc": 1.0,
        }


class _MeanRecordingModel:
    """Module-level (picklable) model recording each fold's train-matrix mean.

    Unlike ``_RecordingModel`` (class-var accumulator, only visible in-process),
    this stores the mean on the instance so it survives joblib/loky pickling in
    the parallel path and is readable via ``fold_results[i]['train_metrics']``.
    """

    def __init__(self, **_kwargs) -> None:
        self.training_metrics: dict = {}

    def train(self, X, y):
        self.training_metrics = {"_train_mean": float(np.asarray(X, dtype=float).mean())}
        return self.training_metrics

    def evaluate(self, X, y):
        return {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0,
            "auc_roc": 1.0,
        }

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.zeros(len(X), dtype=float)


def _cv_inputs():
    # Raw, unscaled features on very different scales; two balanced classes with
    # 4 rows each so 2-fold stratification is feasible.
    X = pd.DataFrame(
        {
            "bp": [100.0, 120.0, 140.0, 160.0, 110.0, 130.0, 150.0, 170.0],
            "chol": [200.0, 250.0, 300.0, 350.0, 210.0, 260.0, 310.0, 360.0],
        }
    )
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    sensitive = pd.DataFrame(index=X.index)
    return X, y, sensitive


def test_cvtrainer_applies_fold_preprocessing_per_fold() -> None:
    # With the factory, each fold's training matrix is standardized on its own
    # rows, so its column means collapse to ~0. Without it, the raw large-scale
    # values are passed through untouched (mean far from 0).
    X, y, sensitive = _cv_inputs()

    _RecordingModel.train_means = []
    CVTrainer(n_folds=2, random_state=0).run_cv_experiment(
        model_class=_RecordingModel, X=X, y=y, sensitive_attrs=sensitive
    )
    raw_means = _RecordingModel.train_means

    _RecordingModel.train_means = []
    CVTrainer(
        n_folds=2, random_state=0, fold_preprocessor_factory=FoldPreprocessor
    ).run_cv_experiment(model_class=_RecordingModel, X=X, y=y, sensitive_attrs=sensitive)
    scaled_means = _RecordingModel.train_means

    assert all(abs(m) > 10 for m in raw_means)  # untouched raw scale
    assert all(abs(m) < 1e-9 for m in scaled_means)  # standardized per fold


def test_cvtrainer_parallel_applies_fold_preprocessing() -> None:
    # Parallel path routes through the picklable _train_fold_worker; assert every
    # fold's train-matrix mean (carried back through fold_results) is standardized.
    X, y, sensitive = _cv_inputs()

    result = CVTrainer(
        n_folds=2, random_state=0, fold_preprocessor_factory=FoldPreprocessor
    ).run_cv_experiment(
        model_class=_MeanRecordingModel,
        X=X,
        y=y,
        sensitive_attrs=sensitive,
        cv_n_jobs=2,
    )
    means = [fr["train_metrics"]["_train_mean"] for fr in result["fold_results"]]
    assert means  # folds actually ran
    assert all(abs(m) < 1e-9 for m in means)


def test_cvtrainer_parallel_without_factory_leaves_raw_scale() -> None:
    X, y, sensitive = _cv_inputs()

    result = CVTrainer(n_folds=2, random_state=0).run_cv_experiment(
        model_class=_MeanRecordingModel,
        X=X,
        y=y,
        sensitive_attrs=sensitive,
        cv_n_jobs=2,
    )
    means = [fr["train_metrics"]["_train_mean"] for fr in result["fold_results"]]
    assert all(abs(m) > 10 for m in means)


def test_get_fold_predictions_applies_fold_preprocessing() -> None:
    # get_fold_predictions reuses one model across folds; after the run its last
    # fold's recorded train mean must be standardized when the factory is set.
    X, y, sensitive = _cv_inputs()

    model = _MeanRecordingModel()
    preds = CVTrainer(
        n_folds=2, random_state=0, fold_preprocessor_factory=FoldPreprocessor
    ).get_fold_predictions(model, X, y, sensitive)
    assert len(preds) == len(X)  # every sample predicted out-of-fold
    assert abs(model.training_metrics["_train_mean"]) < 1e-9

    raw_model = _MeanRecordingModel()
    CVTrainer(n_folds=2, random_state=0).get_fold_predictions(raw_model, X, y, sensitive)
    assert abs(raw_model.training_metrics["_train_mean"]) > 10


def test_apply_fold_preprocessing_helper() -> None:
    # Shared choke point used by the combinatorial mitigation loop: factory=None
    # passes splits through untouched; a factory standardizes val on train stats.
    train = pd.DataFrame({"bp": [10.0, 20.0, 30.0]})
    val = pd.DataFrame({"bp": [20.0]})

    same_train, same_val = apply_fold_preprocessing(None, train, val)
    assert same_train is train and same_val is val

    out_train, out_val = apply_fold_preprocessing(FoldPreprocessor, train, val)
    assert abs(float(out_train["bp"].mean())) < 1e-9  # train standardized on itself
    assert out_val["bp"].iloc[0] == 0.0  # (20 - train_mean 20) / std
