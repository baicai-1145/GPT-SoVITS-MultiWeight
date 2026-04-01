from dataclasses import dataclass
from typing import Any

from gateway.registry import WorkerEndpoint, WorkerRegistry


class WorkerSelectionError(RuntimeError):
    pass


@dataclass
class RoutingDecision:
    target: WorkerEndpoint
    selector: dict[str, Any]


class GatewayRouter:
    def __init__(self, registry: WorkerRegistry):
        self.registry = registry

    def select_worker(self, request_payload: dict[str, Any]) -> RoutingDecision:
        selector = {
            "worker_id": request_payload.get("worker_id"),
            "speaker_id": request_payload.get("speaker_id"),
            "weight_id": request_payload.get("weight_id"),
        }
        target = self.registry.resolve_worker(**selector)
        if target is None:
            raise WorkerSelectionError("no worker matched the request")
        if not target.healthy:
            raise WorkerSelectionError(
                f"matched worker '{target.worker_id}' is unavailable"
                + (f": {target.last_error}" if target.last_error else "")
            )
        return RoutingDecision(target=target, selector=selector)
