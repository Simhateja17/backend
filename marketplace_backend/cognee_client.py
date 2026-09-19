"""The Cognee Cloud boundary.

Only the backend talks to Cognee, with a server-held `X-Api-Key`; the browser and
the model never see a dataset name or a key. Everything sent here has already been
distilled and filtered by `customer_memory`: pseudonymous dataset names, typed facts
and behaviour summaries, never an email, phone number or address.

With `COGNEE_ENABLED` off or no credentials, `CogneeClient.from_env()` returns a
disabled client whose calls raise `CogneeUnavailable`, and every caller treats that
as "run on the cached brief" rather than as a failure.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

import httpx


class CogneeUnavailable(Exception):
    """Cognee is switched off, unconfigured, unreachable, or refused the call."""


class MemoryBackend(Protocol):
    """What the memory layer needs from Cognee. Tests supply a fake."""

    enabled: bool

    async def ensure_dataset(self, name: str) -> str: ...

    async def remember(self, dataset_name: str, texts: list[str], node_set: list[str]) -> dict: ...

    async def recall(self, dataset_name: str, query: str, top_k: int = 8,
                     answer: bool = False, system_prompt: str | None = None) -> list[str]: ...

    async def improve(self, dataset_name: str) -> dict: ...

    async def forget(self, dataset_name: str) -> None: ...


class CogneeClient:
    def __init__(self, base_url: str, api_key: str, *, timeout: float = 30.0,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.enabled = bool(self.base_url and api_key)
        self._headers = {"X-Api-Key": api_key}
        self._timeout = timeout
        self._transport = transport
        self._dataset_ids: dict[str, str] = {}

    @classmethod
    def from_env(cls) -> CogneeClient:
        if os.getenv("COGNEE_ENABLED", "0").lower() not in {"1", "true", "yes"}:
            return cls("", "")
        return cls(os.getenv("COGNEE_SERVICE_URL", ""), os.getenv("COGNEE_API_KEY", ""))

    async def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        if not self.enabled:
            raise CogneeUnavailable("Cognee is not enabled for this deployment")
        try:
            # Cloud answers a slashless collection path with a 307 to the slashed one.
            async with httpx.AsyncClient(base_url=self.base_url, headers=self._headers,
                                         timeout=self._timeout, follow_redirects=True,
                                         transport=self._transport) as client:
                response = await client.request(method, f"/api/v1{path}", **kwargs)
        except httpx.HTTPError as exc:
            raise CogneeUnavailable(f"Cognee unreachable: {type(exc).__name__}") from exc
        if response.status_code >= 400:
            # The body can echo what we sent; the status is enough to act on.
            raise CogneeUnavailable(f"Cognee refused {method} {path}: {response.status_code}")
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError:
            return response.text

    async def ensure_dataset(self, name: str) -> str:
        """Create-or-return by name (the endpoint is idempotent)."""
        if name in self._dataset_ids:
            return self._dataset_ids[name]
        body = await self._call("POST", "/datasets", json={"name": name})
        dataset_id = str((body or {}).get("id", ""))
        if dataset_id:
            self._dataset_ids[name] = dataset_id
        return dataset_id

    async def remember(self, dataset_name: str, texts: list[str], node_set: list[str]) -> dict:
        # A dict of lists is how httpx repeats a form field on an async request.
        form: dict[str, str | list[str]] = {
            "datasetName": dataset_name, "run_in_background": "true",
            "raw_data": list(texts), "node_set": list(node_set)}
        return await self._call("POST", "/remember", data=form) or {}

    async def recall(self, dataset_name: str, query: str, top_k: int = 8,
                     answer: bool = False, system_prompt: str | None = None) -> list[str]:
        """Retrieved context by default; with `answer`, Cognee's completion over it."""
        payload: dict[str, Any] = {
            "query": query, "datasets": [dataset_name], "top_k": top_k,
            "only_context": not answer, "include_references": False,
        }
        if system_prompt:
            payload["system_prompt"] = system_prompt
        body = await self._call("POST", "/recall", json=payload)
        return _texts(body)

    async def improve(self, dataset_name: str) -> dict:
        return await self._call("POST", "/improve", json={
            "dataset_name": dataset_name, "run_in_background": True}) or {}

    async def forget(self, dataset_name: str) -> None:
        await self._call("POST", "/forget", json={"dataset": dataset_name})
        self._dataset_ids.pop(dataset_name, None)


def _texts(body: Any) -> list[str]:
    """Recall answers come back as a list of results whose shape varies by search
    type; keep the text and drop the rest."""
    if body is None:
        return []
    if isinstance(body, str):
        return [body]
    if isinstance(body, dict):
        body = body.get("results", body.get("result", [body]))
    out: list[str] = []
    for item in body if isinstance(body, list) else [body]:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            value = item.get("search_result", item.get("text", item.get("context", item)))
            out.extend(_texts(value) if not isinstance(value, dict) else [json.dumps(value)])
        else:
            out.append(str(item))
    return [text.strip() for text in out if text and text.strip()]
