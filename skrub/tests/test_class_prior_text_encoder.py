import numpy as np
import pandas as pd
import pytest

from skrub import ClassPriorTextEncoder


class DummyClassPriorTextEncoder(ClassPriorTextEncoder):
    def _vectorize(self, column):
        values = np.asarray(column, dtype=object)
        lengths = np.asarray([0.0 if v is None else float(len(str(v))) for v in values])
        return np.column_stack([lengths, lengths + 1.0])


def test_class_prior_requires_matching_length():
    enc = DummyClassPriorTextEncoder(n_components=None, train_projection_head=False)
    x = pd.Series(["a", "bb", "ccc"])
    y = ["A", "B"]

    with pytest.raises(ValueError, match="`y` has length 2 but expected 3."):
        enc.fit_transform(x, y=y)


def test_class_prior_applies_on_fit_transform():
    enc = DummyClassPriorTextEncoder(
        n_components=None, prior_strength=1.0, train_projection_head=False
    )
    x = pd.Series(["a", "bb", "ccc", "dddd"])
    y = ["A", "A", "B", "B"]

    out = enc.fit_transform(x, y=y).to_numpy()
    assert np.allclose(out[0], out[1])
    assert np.allclose(out[2], out[3])
    assert not np.allclose(out[0], out[2])


def test_class_prior_applies_on_transform():
    enc = DummyClassPriorTextEncoder(
        n_components=None, prior_strength=1.0, train_projection_head=False
    )
    x_train = pd.Series(["a", "bb", "ccc", "dddd"])
    y_train = ["A", "A", "B", "B"]
    enc.fit(x_train, y=y_train)
    # Avoid triggering model loading in TextEncoder.transform during this unit test.
    enc.__dict__["_estimator"] = object()

    x_test = pd.Series(["x", "yy"])
    y_test = ["B", "A"]
    out = enc.transform(x_test, class_categories=y_test).to_numpy()

    assert np.allclose(out[0], enc.class_priors_["B"])
    assert np.allclose(out[1], enc.class_priors_["A"])


def test_projection_output_dim_requires_y():
    enc = DummyClassPriorTextEncoder(
        n_components=None,
        projection_output_dim=1,
        train_projection_head=True,
    )
    x = pd.Series(["a", "bb", "ccc"])
    with pytest.raises(ValueError, match="requires y during fit"):
        enc.fit_transform(x, y=None)


def test_projection_output_dim_requires_training_head():
    enc = DummyClassPriorTextEncoder(
        n_components=None,
        projection_output_dim=1,
        train_projection_head=False,
    )
    x = pd.Series(["a", "bb", "ccc"])
    y = ["A", "A", "B"]
    with pytest.raises(ValueError, match="train_projection_head=False"):
        enc.fit_transform(x, y=y)


def test_projection_output_dim_changes_output_width():
    pytest.importorskip("torch")
    enc = DummyClassPriorTextEncoder(
        n_components=None,
        projection_output_dim=1,
        train_projection_head=True,
        projection_n_epochs=2,
    )
    x = pd.Series(["a", "bb", "ccc", "dddd"])
    y = ["A", "A", "B", "B"]
    out = enc.fit_transform(x, y=y)
    assert out.shape[1] == 1
