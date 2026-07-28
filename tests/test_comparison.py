from __future__ import annotations

import numpy as np
import pytest

from crisisforge.evaluation.comparison import paired_block_bootstrap_mean


def test_paired_block_bootstrap_is_deterministic_and_directional() -> None:
    differences = np.linspace(-0.2, -0.1, 40)
    first = paired_block_bootstrap_mean(
        differences,
        replications=500,
        block_length=4,
        confidence_level=0.95,
        random_seed=7,
    )
    second = paired_block_bootstrap_mean(
        differences,
        replications=500,
        block_length=4,
        confidence_level=0.95,
        random_seed=7,
    )
    assert first == second
    assert first["ci_upper"] < 0.0


@pytest.mark.parametrize(
    ("replications", "block_length"),
    [(99, 2), (100, 0), (100, 10)],
)
def test_paired_block_bootstrap_rejects_invalid_design(
    replications: int,
    block_length: int,
) -> None:
    with pytest.raises(ValueError):
        paired_block_bootstrap_mean(
            np.array([0.1, 0.2, 0.3]),
            replications=replications,
            block_length=block_length,
            confidence_level=0.95,
            random_seed=1,
        )
