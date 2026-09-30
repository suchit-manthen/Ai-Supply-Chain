import numpy as np
import pytest

from data import generate
from models import FEATS, _features


@pytest.fixture(scope="module")
def sample():
    df, _, _ = generate()
    return df[df["product"] == "soda"].sort_values("date").reset_index(drop=True)


def test_feature_columns_and_length(sample):
    x = _features(sample)
    assert list(x.columns) == FEATS
    assert len(x) == len(sample)
    assert not x.isna().any().any()


def test_calendar_features_correct(sample):
    x = _features(sample)
    assert (x["dow"] == sample.date.dt.dayofweek).all()
    assert (x["is_weekend"] == (sample.date.dt.dayofweek >= 5).astype(int)).all()
    assert (x["month"] == sample.date.dt.month).all()
    assert (x["doy"] == sample.date.dt.dayofyear).all()


def test_exogenous_features_copied_verbatim(sample):
    x = _features(sample)
    for c in ["discount", "price", "temp", "holiday_lift"]:
        assert np.allclose(x[c].to_numpy(), sample[c].to_numpy())


def test_index_preserved(sample):
    x = _features(sample)
    assert x.index.equals(sample.index)
