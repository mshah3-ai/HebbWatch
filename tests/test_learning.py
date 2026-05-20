import numpy as np

from hebbwatch.core.learning_rules import HebbianRule, OjaRule, build_learning_rule


def test_hebbian_delta_shape():
    rule = HebbianRule()
    weights = np.zeros((3, 2))
    delta = rule.delta(weights, np.ones(2), np.ones(3))
    assert delta.shape == weights.shape
    assert delta.mean() > 0


def test_oja_rule_shape():
    rule = OjaRule()
    weights = np.ones((4, 3)) * 0.1
    delta = rule.delta(weights, np.ones(3), np.ones(4))
    assert delta.shape == weights.shape


def test_rule_builder():
    assert build_learning_rule("competitive", 0.01).name == "competitive"
