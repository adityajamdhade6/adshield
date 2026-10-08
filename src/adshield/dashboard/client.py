"""HTTP client for the AdShield scoring API, with errors a dashboard user can act on."""

from __future__ import annotations

from typing import Any

import httpx

from adshield.api.schemas import ClickRequest

DETAIL_TOP_K = 11  # every engineered feature, so the detail view shows the full breakdown


class ApiError(RuntimeError):
    """The API could not be used. The message is safe to show to an end user."""


class ScoringClient:
    def __init__(self, base_url: str, timeout: float = 120.0, http_client: httpx.Client | None = None):
        self._http = http_client or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return self._http.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            raise ApiError(
                "Cannot reach the scoring service. Check that it is running and the address is right."
            ) from exc
        except httpx.TimeoutException as exc:
            raise ApiError("The scoring service took too long to answer. Try a smaller file.") from exc

    def health(self) -> dict[str, Any]:
        resp = self._request("GET", "/health")
        if resp.status_code != 200:
            raise ApiError(f"Scoring service health check failed (HTTP {resp.status_code}).")
        return resp.json()

    def score_batch(self, records: list[ClickRequest]) -> list[dict[str, Any]]:
        """Score records in the given order without recording them as live IP history."""
        payload = [r.model_dump(mode="json") for r in records]
        resp = self._request(
            "POST",
            "/score/batch",
            json=payload,
            params={"commit_history": "false", "top_k": DETAIL_TOP_K},
        )
        if resp.status_code == 422:
            first = resp.json()["detail"][0]
            where = ".".join(str(p) for p in first["loc"][1:])
            raise ApiError(f"The scoring service rejected the data ({where}: {first['msg']}).")
        if resp.status_code != 200:
            raise ApiError(f"The scoring service failed (HTTP {resp.status_code}). Try again later.")
        results = resp.json()["results"]
        if len(results) != len(records):
            raise ApiError("The scoring service returned a different number of results than were sent.")
        return results
