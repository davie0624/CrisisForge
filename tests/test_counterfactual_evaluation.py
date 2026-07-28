from __future__ import annotations

import numpy as np

from crisisforge.evaluation.counterfactual import (
    _misspecified_models,
    build_registered_interventions,
)


def _configuration() -> dict[str, object]:
    return {
        "policy_intervention": {
            "active_steps": 3,
            "factual_control_value": 0.0,
            "treatment_value": 0.5,
        },
        "controlled_direct_effect": {
            "mediator_value": 0.0,
            "mediators": ["yield", "liquidity", "credit"],
            "active_steps": 5,
        },
        "misspecification": {
            "reduced_yield_policy_multiplier": 0.5,
            "remove_lagged_equity_feedback": True,
        },
    }


def test_controlled_effect_interventions_hold_identical_mediators() -> None:
    total, reference, treatment = build_registered_interventions(_configuration())
    assert total.effect_type == "total"
    assert reference.effect_type == "controlled"
    assert treatment.effect_type == "controlled"
    assert reference.controlled == treatment.controlled
    assert reference.policy != treatment.policy
    assert all(
        np.isclose(value, 0.0)
        for schedule in reference.controlled.values()
        for value in schedule.values()
    )


def test_misspecified_models_change_registered_structural_coefficients() -> None:
    from crisisforge.counterfactual import SCMParameters

    truth = SCMParameters()
    models = _misspecified_models(truth, _configuration())
    assert np.isclose(
        models["reduced_yield_transmission"].parameters.yield_policy,
        truth.yield_policy * 0.5,
    )
    assert (
        models["no_lagged_equity_feedback"].parameters.policy_equity_feedback
        == 0.0
    )
