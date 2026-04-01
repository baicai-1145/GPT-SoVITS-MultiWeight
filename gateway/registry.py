import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
import yaml


@dataclass
class WorkerEndpoint:
    worker_id: str
    speaker_id: str | None
    weight_id: str | None
    base_url: str
    timeout_seconds: float
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    healthy: bool = False
    last_health: dict[str, Any] | None = None
    last_error: str | None = None
    last_checked_at: float | None = None

    def route_keys(self) -> set[str]:
        keys = {self.worker_id}
        for value in [self.speaker_id, self.weight_id]:
            if value:
                keys.add(value)
        return keys


class WorkerRegistry:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self._lock = threading.RLock()
        self.gateway_config = self._load_config(self.config_path)
        self.request_timeout_seconds = float(self.gateway_config["gateway"].get("request_timeout_seconds", 180.0))
        self.healthcheck_interval_seconds = float(
            self.gateway_config["gateway"].get("healthcheck_interval_seconds", 10.0)
        )
        self.default_worker_timeout_seconds = float(
            self.gateway_config["gateway"].get("default_worker_timeout_seconds", 180.0)
        )
        self._workers = self._build_workers(self.gateway_config.get("workers", []))
        self._stop_event = threading.Event()
        self._poller = None

    @staticmethod
    def _load_config(config_path: Path) -> dict:
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    def _build_workers(self, worker_items: list[dict]) -> list[WorkerEndpoint]:
        workers = []
        for item in worker_items:
            worker_id = item["worker_id"]
            base_url = str(item["base_url"]).rstrip("/")
            workers.append(
                WorkerEndpoint(
                    worker_id=worker_id,
                    speaker_id=item.get("speaker_id"),
                    weight_id=item.get("weight_id") or item.get("speaker_id") or worker_id,
                    base_url=base_url,
                    timeout_seconds=float(item.get("timeout_seconds", self.default_worker_timeout_seconds)),
                    tags=list(item.get("tags", []) or []),
                    metadata=dict(item.get("metadata", {}) or {}),
                )
            )
        return workers

    def start_health_polling(self):
        if self._poller is not None:
            return

        def loop():
            while not self._stop_event.is_set():
                self.refresh_health()
                self._stop_event.wait(self.healthcheck_interval_seconds)

        self.refresh_health()
        self._poller = threading.Thread(target=loop, name="gateway-worker-health", daemon=True)
        self._poller.start()

    def stop_health_polling(self):
        self._stop_event.set()
        if self._poller is not None:
            self._poller.join(timeout=2)

    def refresh_health(self):
        for worker in self._workers:
            self._refresh_worker_health(worker)

    def _refresh_worker_health(self, worker: WorkerEndpoint):
        url = f"{worker.base_url}/health"
        checked_at = time.time()
        try:
            response = requests.get(url, timeout=min(worker.timeout_seconds, 10.0))
            payload = response.json()
            healthy = response.status_code == 200 and bool(payload.get("ready", False))
            last_error = None if healthy else f"health status={response.status_code}"
        except Exception as exc:
            payload = None
            healthy = False
            last_error = str(exc)

        with self._lock:
            worker.healthy = healthy
            worker.last_health = payload
            worker.last_error = last_error
            worker.last_checked_at = checked_at

    def resolve_worker(self, worker_id: str | None = None, speaker_id: str | None = None, weight_id: str | None = None):
        target_keys = [value for value in [worker_id, speaker_id, weight_id] if value not in [None, ""]]
        with self._lock:
            candidates = list(self._workers)
            if target_keys:
                candidates = [worker for worker in candidates if any(key in worker.route_keys() for key in target_keys)]
            healthy_candidates = [worker for worker in candidates if worker.healthy]
            if healthy_candidates:
                return healthy_candidates[0]
            if candidates:
                return candidates[0]
        return None

    def list_workers(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "worker_id": worker.worker_id,
                    "speaker_id": worker.speaker_id,
                    "weight_id": worker.weight_id,
                    "base_url": worker.base_url,
                    "timeout_seconds": worker.timeout_seconds,
                    "healthy": worker.healthy,
                    "last_error": worker.last_error,
                    "last_checked_at": worker.last_checked_at,
                    "tags": list(worker.tags),
                    "metadata": dict(worker.metadata),
                }
                for worker in self._workers
            ]

    def gateway_meta(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "request_timeout_seconds": self.request_timeout_seconds,
            "healthcheck_interval_seconds": self.healthcheck_interval_seconds,
            "workers": self.list_workers(),
        }
