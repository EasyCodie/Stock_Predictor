from typing import Any
import numpy as np

class CalibratedClassifierCV:
    def fit(
        self,
        X: np.ndarray[Any, Any],
        y: np.ndarray[Any, Any],
        sample_weight: np.ndarray[Any, Any] | None = ...,
        **fit_params: dict[str, Any],
    ) -> "CalibratedClassifierCV": ...
