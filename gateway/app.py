import argparse
import os
import sys
import traceback
from typing import Any

import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from gateway.registry import WorkerRegistry
from gateway.router import GatewayRouter, WorkerSelectionError


def create_gateway_app(registry: WorkerRegistry, router: GatewayRouter) -> FastAPI:
    app = FastAPI(title="GPT-SoVITS Gateway")

    @app.on_event("startup")
    async def startup():
        registry.start_health_polling()

    @app.on_event("shutdown")
    async def shutdown():
        registry.stop_health_polling()

    def sanitize_forward_payload(payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        for key in ["worker_id", "speaker_id", "weight_id"]:
            payload.pop(key, None)
        return payload

    def proxy_to_worker(payload: dict[str, Any]):
        decision = router.select_worker(payload)
        worker = decision.target
        forwarded_payload = sanitize_forward_payload(payload)
        response = requests.post(
            f"{worker.base_url}/tts",
            json=forwarded_payload,
            timeout=worker.timeout_seconds,
            stream=True,
        )
        if response.status_code != 200:
            content_type = response.headers.get("content-type", "")
            body = response.json() if "application/json" in content_type else {"message": response.text}
            return JSONResponse(
                status_code=response.status_code,
                content={
                    "message": "worker request failed",
                    "worker_id": worker.worker_id,
                    "worker_base_url": worker.base_url,
                    "worker_response": body,
                },
            )

        headers = {}
        media_type = response.headers.get("content-type", "application/octet-stream")
        if response.headers.get("content-length"):
            headers["content-length"] = response.headers["content-length"]
        headers["x-gateway-worker-id"] = worker.worker_id
        if "audio/" in media_type:
            return StreamingResponse(
                response.iter_content(chunk_size=8192),
                media_type=media_type,
                headers=headers,
            )
        body = response.content
        return Response(content=body, media_type=media_type, headers=headers)

    @app.get("/health")
    async def health():
        workers = registry.list_workers()
        ready = any(worker["healthy"] for worker in workers)
        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "ready": ready,
                "workers_total": len(workers),
                "workers_healthy": sum(1 for worker in workers if worker["healthy"]),
                "gateway": registry.gateway_meta(),
            },
        )

    @app.get("/meta")
    async def meta():
        return JSONResponse(status_code=200, content=registry.gateway_meta())

    @app.get("/workers")
    async def workers():
        return JSONResponse(status_code=200, content={"workers": registry.list_workers()})

    @app.get("/tts")
    async def tts_get(request: Request):
        try:
            return proxy_to_worker(dict(request.query_params))
        except WorkerSelectionError as exc:
            return JSONResponse(status_code=503, content={"message": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"message": "gateway proxy failed", "exception": str(exc)})

    @app.post("/tts")
    async def tts_post(request: Request):
        payload = await request.json()
        try:
            return proxy_to_worker(payload)
        except WorkerSelectionError as exc:
            return JSONResponse(status_code=503, content={"message": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"message": "gateway proxy failed", "exception": str(exc)})

    return app


def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS gateway")
    parser.add_argument(
        "-c",
        "--gateway-config",
        type=str,
        default="configs/gateway/routes.yaml",
        help="Path to gateway route config yaml",
    )
    parser.add_argument("-a", "--bind-addr", type=str, default="127.0.0.1")
    parser.add_argument("-p", "--port", type=int, default=9880)
    args = parser.parse_args()

    registry = WorkerRegistry(args.gateway_config)
    app = create_gateway_app(registry, GatewayRouter(registry))

    try:
        host = None if args.bind_addr == "None" else args.bind_addr
        uvicorn.run(app=app, host=host, port=args.port, workers=1)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
