from datetime import datetime
from typing import Any
import pandas as pd

def download(
    tickers: str | list[str],
    start: str | datetime | None = ...,
    end: str | datetime | None = ...,
    auto_adjust: bool = ...,
    back_adjust: bool = ...,
    repair: bool = ...,
    progress: bool = ...,
    group_by: str = ...,
    threads: bool = ...,
    **kwargs: Any
) -> pd.DataFrame | None: ...
