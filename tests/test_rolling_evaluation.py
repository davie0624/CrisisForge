from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crisisforge.evaluation.rolling import (
    _origin_positions,
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


def test_registered_origin_builder_rejects_test_split() -> None:
    dates = pd.date_range("2010-01-04", periods=100, freq="B")
    with pytest.raises(ValueError, match="test split sealed"):
        _origin_positions(
            dates,
            evaluation_split="test",
            train_end=dates[39].date().isoformat(),
            validation_end=dates[79].date().isoformat(),
            horizon=5,
            stride=5,
        )
