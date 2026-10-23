"""Metrics are checked against sklearn and against hand-computed values."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.utils.metrics import (N_BINS, PixelHistogram, classification_metrics, dice_score, expected_calibration_error,
                               fit_temperature, optimise_thresholds, qwk, scores_to_grades, softmax)


@pytest.fixture
def pixels() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A noisy 'vessel map': probabilities correlated with the target, plus a FOV that hides a border."""
    rng = np.random.default_rng(0)
    target = (rng.random((128, 128)) < 0.12).astype(np.uint8)
    prob = np.clip(0.6 * target + rng.normal(0.25, 0.18, target.shape), 0, 1).astype(np.float32)
    fov = np.zeros_like(target, dtype=bool)
    fov[8:-8, 8:-8] = True
    return prob, target, fov


def test_histogram_auc_matches_sklearn(pixels):
    prob, target, fov = pixels
    hist = PixelHistogram()
    hist.update(prob, target, fov)
    s = hist.summary(0.5)
    assert s["auc_roc"] == pytest.approx(roc_auc_score(target[fov], prob[fov]), abs=1e-3)
    assert s["auc_pr"] == pytest.approx(average_precision_score(target[fov], prob[fov]), abs=2e-3)
    assert s["n_pixels"] == fov.sum()
    assert s["vessel_fraction"] == pytest.approx(target[fov].mean())


def test_histogram_accumulates_over_images(pixels):
    prob, target, fov = pixels
    one, two = PixelHistogram(), PixelHistogram()
    one.update(prob, target, fov)
    two.update(prob[:64], target[:64], fov[:64])
    two.update(prob[64:], target[64:], fov[64:])
    np.testing.assert_array_equal(one.pos, two.pos)
    np.testing.assert_array_equal(one.neg, two.neg)


def test_histogram_dice_matches_thresholded_mask(pixels):
    prob, target, fov = pixels
    hist = PixelHistogram()
    hist.update(prob, target, fov)
    thr = 0.5  # on a bin edge, so the histogram threshold is exact
    s = hist.summary(thr)
    pred = prob >= thr
    assert s["dice"] == pytest.approx(dice_score(pred, target, fov), abs=1e-9)
    tp = (pred & (target == 1))[fov].sum()
    assert s["sensitivity"] == pytest.approx(tp / target[fov].sum())


def test_best_threshold_is_argmax_of_dice(pixels):
    prob, target, fov = pixels
    hist = PixelHistogram()
    hist.update(prob, target, fov)
    thr, best = hist.best_threshold()
    assert 0 <= thr < 1
    assert hist.summary(thr)["dice"] == pytest.approx(best)
    assert best >= max(hist.summary(t)["dice"] for t in np.linspace(0, 1, 41)[:-1]) - 1e-12


def test_histogram_ignores_pixels_outside_fov(pixels):
    prob, target, fov = pixels
    a, b = PixelHistogram(), PixelHistogram()
    a.update(prob, target, fov)
    b.update(np.where(fov, prob, 1 - prob), np.where(fov, target, 1 - target), fov)  # flip everything outside the FOV
    np.testing.assert_array_equal(a.pos, b.pos)
    np.testing.assert_array_equal(a.neg, b.neg)


def test_histogram_probability_one_lands_in_last_bin():
    hist = PixelHistogram()
    hist.update(np.array([[1.0, 0.0]]), np.array([[1, 0]]), np.ones((1, 2), bool))
    assert hist.pos[N_BINS - 1] == 1 and hist.neg[0] == 1


def test_dice_score_edge_cases():
    fov = np.ones((4, 4), bool)
    m = np.eye(4, dtype=np.uint8)
    assert dice_score(m.astype(bool), m, fov) == pytest.approx(1.0)
    assert dice_score(np.zeros_like(m, bool), m, fov) == pytest.approx(0.0)


def test_qwk():
    y = np.array([0, 1, 2, 3, 4, 2, 1])
    assert qwk(y, y) == pytest.approx(1.0)
    # an off-by-one error is penalised less than an off-by-four error
    near, far = y.copy(), y.copy()
    near[0], far[0] = 1, 4
    assert qwk(y, near) > qwk(y, far)


def test_softmax_and_temperature():
    logits = np.array([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    p = softmax(logits)
    np.testing.assert_allclose(p.sum(1), 1.0)
    np.testing.assert_allclose(p[1], 1 / 3)
    assert softmax(logits, 5.0)[0].max() < p[0].max()  # higher temperature = flatter
    assert np.isfinite(softmax(np.array([[1e4, 0.0]]))).all()


def test_fit_temperature_recovers_overconfidence():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 1.5, (20000, 5))
    labels = np.array([rng.choice(5, p=row) for row in softmax(z)])
    assert fit_temperature(3.0 * z, labels) == pytest.approx(3.0, rel=0.1)  # logits 3x too sharp -> T ~ 3


def test_expected_calibration_error():
    conf = np.full(100, 0.9)
    assert expected_calibration_error(conf, np.r_[np.ones(90), np.zeros(10)])[0] == pytest.approx(0.0)
    assert expected_calibration_error(conf, np.zeros(100))[0] == pytest.approx(0.9)
    ece, bins = expected_calibration_error(np.array([0.0, 1.0]), np.array([0.0, 1.0]), n_bins=10)
    assert bins["count"].sum() == 2 and ece == pytest.approx(0.0)


def test_scores_to_grades():
    t = np.array([0.5, 1.5, 2.5, 3.5])
    np.testing.assert_array_equal(scores_to_grades(np.array([-1, 0.4, 0.6, 2.0, 3.49, 9.0]), t), [0, 0, 1, 2, 3, 4])
    np.testing.assert_array_equal(scores_to_grades(np.array([1.0]), t[::-1]), [1])  # order of cut points does not matter


def test_optimise_thresholds_not_worse_than_default():
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 5, 400)
    scores = labels * 0.7 + rng.normal(0, 0.4, 400)  # compressed scale: default cut points are poor
    default = np.array([0.5, 1.5, 2.5, 3.5])
    t = optimise_thresholds(scores, labels)
    assert np.all(np.diff(t) >= 0)
    assert qwk(labels, scores_to_grades(scores, t)) >= qwk(labels, scores_to_grades(scores, default))


def test_classification_metrics_referable():
    y_true = np.array([0, 1, 2, 3, 4, 0, 2, 4])
    y_pred = np.array([0, 2, 2, 1, 4, 0, 2, 4])  # one false referral (1 -> 2), one missed referral (3 -> 1)
    score = (y_pred >= 2).astype(float)
    m = classification_metrics(y_true, y_pred, score)
    assert m["referable_sensitivity"] == pytest.approx(4 / 5)
    assert m["referable_specificity"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["confusion_matrix"].shape == (5, 5) and m["confusion_matrix"].sum() == 8
