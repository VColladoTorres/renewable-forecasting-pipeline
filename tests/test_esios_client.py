
# MIT © 2025 MSc Candidate
"""Unit test for EsiosClient using responses to mock HTTP."""
import json
from datetime import datetime, timezone

import pandas as pd
import pytest
import responses

from src.data.esios_client import EsiosClient


@responses.activate
def test_fetch_series():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 1, tzinfo=timezone.utc)
    sample = {
        "data": [
            {
                "attributes": {
                    "datetime": "2024-01-01T00:00:00Z",
                    "value": 500,
                }
            },
            {
                "attributes": {
                    "datetime": "2024-01-01T01:00:00Z",
                    "value": 600,
                }
            },
        ],
        "links": {"next": None},
    }
    responses.add(
        responses.GET,
        "https://api.esios.ree.es/archives/75",
        json=sample,
        status=200,
    )

    client = EsiosClient(token="dummy")
    df = client.fetch_series(start, end, technology="wind")
    assert isinstance(df, pd.DataFrame)
    assert df["mw"].iloc[0] == 500
