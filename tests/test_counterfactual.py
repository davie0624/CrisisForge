from __future__ import annotations

import numpy as np

from crisisforge.counterfactual import (
    Intervention,
    RegimeSwitchingSCM,
    SCMExogenous,
    estimate_causal_effect,
    evaluate_counterfactual_error,
)


def test_seed_reproduces_full_exogenous_and_factual_paths() -> None:
    scm = RegimeSwitchingSCM()
    first_noise = scm.generate_exogenous(num_paths=64, horizon=12, seed=17)
    second_noise = scm.generate_exogenous(num_paths=64, horizon=12, seed=17)
    assert np.array_equal(first_noise.innovations, second_noise.innovations)
    assert np.array_equal(first_noise.regime_uniforms, second_noise.regime_uniforms)
    first = scm.simulate(first_noise)
    second = scm.simulate(second_noise)
    assert np.array_equal(first.as_array(), second.as_array())
    assert np.array_equal(first.regime, second.regime)


def test_paired_simulation_uses_identical_regime_path() -> None:
    scm = RegimeSwitchingSCM()
    intervention = Intervention(policy={0: 0.50, 1: 0.50}, label="policy shock")
    paired = scm.paired_simulation(
        num_paths=128,
        horizon=15,
        intervention=intervention,
        seed=123,
    )
    assert np.array_equal(paired.factual.regime, paired.counterfactual.regime)
    assert np.allclose(paired.counterfactual.policy[:, :2], 0.50)
    assert not np.allclose(
        paired.factual.equity,
        paired.counterfactual.equity,
    )


def test_lagged_equity_feedback_is_time_unrolled() -> None:
    scm = RegimeSwitchingSCM()
    intervention = Intervention(policy={0: 0.50})
    paired = scm.paired_simulation(
        num_paths=32,
        horizon=3,
        intervention=intervention,
        seed=41,
    )
    policy_difference_t1 = paired.counterfactual.policy[:, 1] - paired.factual.policy[:, 1]
    expected = scm.parameters.policy_ar * (
        paired.counterfactual.policy[:, 0] - paired.factual.policy[:, 0]
    ) + scm.parameters.policy_equity_feedback * (
        paired.counterfactual.equity[:, 0] - paired.factual.equity[:, 0]
    )
    assert np.allclose(policy_difference_t1, expected)


def test_abduction_action_prediction_recovers_direct_counterfactual() -> None:
    scm = RegimeSwitchingSCM()
    intervention = Intervention(
        policy={time: 0.40 for time in range(4)},
        label="paired policy path",
    )
    paired = scm.paired_simulation(
        num_paths=256,
        horizon=20,
        intervention=intervention,
        seed=2026,
    )
    recovered = scm.abduction_action_prediction(paired.factual, intervention)
    assert np.allclose(
        recovered.as_array(),
        paired.counterfactual.as_array(),
        atol=2e-12,
    )
    assert np.array_equal(recovered.regime, paired.counterfactual.regime)


def test_total_and_controlled_effects_are_labeled_and_distinct() -> None:
    scm = RegimeSwitchingSCM()
    noise = scm.generate_exogenous(num_paths=512, horizon=20, seed=88)
    factual = scm.simulate(noise)
    policy_path = {time: 0.50 for time in range(5)}
    total_intervention = Intervention(policy=policy_path)
    controlled_intervention = Intervention(
        policy=policy_path,
        controlled={
            "yield": {time: 0.0 for time in range(20)},
            "liquidity": {time: 0.0 for time in range(20)},
            "credit": {time: 0.0 for time in range(20)},
        },
    )
    total_paths = scm.simulate(noise, total_intervention)
    controlled_paths = scm.simulate(noise, controlled_intervention)
    total = estimate_causal_effect(
        factual,
        total_paths,
        effect_type=total_intervention.effect_type,
    )
    controlled = estimate_causal_effect(
        factual,
        controlled_paths,
        effect_type=controlled_intervention.effect_type,
    )
    assert total.effect_type == "total"
    assert controlled.effect_type == "controlled"
    assert not np.allclose(total.mean_path, controlled.mean_path)


def test_counterfactual_metrics_are_zero_at_ground_truth() -> None:
    scm = RegimeSwitchingSCM()
    intervention = Intervention(policy={time: 0.35 for time in range(3)})
    paired = scm.paired_simulation(
        num_paths=256,
        horizon=15,
        intervention=intervention,
        seed=7,
    )
    error = evaluate_counterfactual_error(
        factual=paired.factual,
        estimated_counterfactual=paired.counterfactual,
        ground_truth_counterfactual=paired.counterfactual,
        effect_type=intervention.effect_type,
    )
    assert error.ate_error == 0.0
    assert error.path_rmse == 0.0
    assert error.tail_effect_error == 0.0


def test_tail_direction_is_outcome_specific() -> None:
    scm = RegimeSwitchingSCM()
    intervention = Intervention(policy={time: 0.50 for time in range(5)})
    paired = scm.paired_simulation(
        num_paths=512,
        horizon=20,
        intervention=intervention,
        seed=92,
    )
    equity = estimate_causal_effect(
        paired.factual,
        paired.counterfactual,
        outcome="equity",
        loss_sign=-1.0,
    )
    volatility = estimate_causal_effect(
        paired.factual,
        paired.counterfactual,
        outcome="volatility",
        loss_sign=1.0,
    )
    opposite_volatility = estimate_causal_effect(
        paired.factual,
        paired.counterfactual,
        outcome="volatility",
        loss_sign=-1.0,
    )
    assert np.isfinite(equity.tail_effect)
    assert np.isfinite(volatility.tail_effect)
    assert not np.isclose(
        volatility.tail_effect,
        opposite_volatility.tail_effect,
    )


def test_zero_tail_loss_sign_is_rejected() -> None:
    scm = RegimeSwitchingSCM()
    paths = scm.simulate(scm.generate_exogenous(num_paths=8, horizon=5, seed=9))
    with np.testing.assert_raises_regex(ValueError, "finite and nonzero"):
        estimate_causal_effect(paths, paths, loss_sign=0.0)


def test_controlled_mediators_follow_declared_within_slice_order() -> None:
    scm = RegimeSwitchingSCM()
    exogenous = SCMExogenous(
        innovations=np.zeros((1, 1, 6)),
        regime_uniforms=np.zeros((1, 1)),
        initial_regimes=np.zeros(1, dtype=int),
    )
    controlled = Intervention(
        policy={},
        controlled={"yield": {0: 1.0}},
    )
    path = scm.simulate(exogenous, controlled)
    expected_liquidity = scm.parameters.liquidity_yield
    expected_credit = (
        scm.parameters.credit_yield + scm.parameters.credit_liquidity * expected_liquidity
    )
    assert np.isclose(path.yield_rate[0, 0], 1.0)
    assert np.isclose(path.liquidity[0, 0], expected_liquidity)
    assert np.isclose(path.credit[0, 0], expected_credit)


def test_effect_metrics_reject_unpaired_regime_paths() -> None:
    scm = RegimeSwitchingSCM()
    first = scm.simulate(scm.generate_exogenous(num_paths=8, horizon=5, seed=1))
    second = scm.simulate(scm.generate_exogenous(num_paths=8, horizon=5, seed=2))
    with np.testing.assert_raises_regex(
        ValueError,
        "identical exogenous regime paths",
    ):
        estimate_causal_effect(first, second)


def test_exogenous_regime_labels_must_be_integer() -> None:
    scm = RegimeSwitchingSCM()
    invalid = SCMExogenous(
        innovations=np.zeros((1, 2, 6)),
        regime_uniforms=np.zeros((1, 2)),
        initial_regimes=np.array([0.5]),
    )
    with np.testing.assert_raises_regex(ValueError, "integer labels"):
        scm.simulate(invalid)
