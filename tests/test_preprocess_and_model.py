import numpy as np
import pytest
import torch

from seizure_detector.models import MODELS, create_model
from seizure_detector.preprocess import BIPOLAR_PAIRS, canonical_channel, robust_scale


def test_channel_normalization_and_montage():
    assert canonical_channel("EEG Fp1-REF") == "FP1"
    assert canonical_channel("T7") == "T3"
    assert len(BIPOLAR_PAIRS) == 18


def test_robust_scaling():
    rng = np.random.default_rng(3)
    data = rng.normal(size=(18, 1280))
    scaled = robust_scale(data)
    assert scaled.shape == (18, 1280)
    assert scaled.dtype == np.float32
    assert np.max(np.abs(scaled)) <= 8


def test_eegnet_output_shape():
    model = create_model("eegnet", channels=18, samples=1280)
    x = torch.randn(4, 18, 1280)
    mask = torch.ones(4, 18)
    logits = model(x, mask)
    assert logits.shape == (4,)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_all_registered_models_produce_expected_shape(name):
    model = create_model(name, channels=18, samples=1280, dropout=0.2)
    model.eval()
    x = torch.randn(3, 18, 1280)
    mask = torch.ones(3, 18)
    logits = model(x, mask)
    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()


def test_unknown_model_name_raises():
    with pytest.raises(ValueError):
        create_model("not-a-real-model", channels=18, samples=1280)


def test_eeg_conformer_capacity_and_bandpower():
    model = create_model("eeg_conformer", channels=18, samples=1280, dropout=0.2)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 100_000
    model.eval()
    x = torch.randn(2, 18, 1280)
    logits = model(x, torch.ones(2, 18))
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
