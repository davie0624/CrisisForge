from __future__ import annotations

import numpy as np

from crisisforge.evaluation.rolling import (
    build_generator,
    rolling_cumulative_returns,
)


def test_rolling_cumulative_returns_uses_complete_horizon() -> None:
    returns = np.array(
        [
            [0.10, 0.00],
            [-0.05, 0.02],
            [0.01, -0.03],
        ]
    )
    result = rolling_cumulative_returns(returns, horizon=2)
    assert result.shape == (2, 2)
    assert np.allclose(result[0], [0.045, 0.02])
    assert np.allclose(
        result[1],
        [
            (1.0 - 0.05) * (1.0 + 0.01) - 1.0,
            (1.0 + 0.02) * (1.0 - 0.03) - 1.0,
        ],
    )


def test_generator_factory_builds_configured_model() -> None:
    generator = build_generator(
        {
            "id": "block",
            "kind": "moving_block",
            "block_length": 5,
        }
    )
    assert generator.block_length == 5
