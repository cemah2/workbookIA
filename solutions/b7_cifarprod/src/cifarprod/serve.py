"""The prediction API of cifarprod: FastAPI when it is installed, else the standard library (http.server).

    app = create_app("artifact")                      # FastAPI; or: uvicorn --factory cifarprod.serve:app_from_env
    server = make_server("artifact", port=0)          # the fallback: a ThreadingHTTPServer with the same routes
    with running("artifact") as url:                  # in a thread, on a free port, always stopped at the end
        requests.post(f"{url}/predict", json={"pixels": image.tolist()})
    python -m cifarprod.serve --artifact artifact --port 8000

Routes: GET /health -> {"status": "ok", "package_version", "model": {"arch", "widths", "classes", "weights_sha256",
"data_sha256", "created"}, "torch"}; POST /predict {"pixels": 32 x 32 x 3 integers in 0..255, RGB} -> {"label",
"probability", "model_version"}; POST /predict_batch {"images": [1 to 64 such images]} -> {"predictions": [{"label",
"probability"}, ...], "model_version"}. A body that does not follow the schema: 422 {"detail": [{"loc", "msg",
"type"}]}, without the input (it may be personal data); an error of the model: 500 {"detail": "internal error"}.
Both servers validate with the same pydantic models and predict with the same Service (artifact.predict_images, the
preprocessing of the training).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from pydantic import BaseModel, ValidationError, field_validator

from . import __version__
from .artifact import load_artifact, predict_images

log = logging.getLogger(__name__)
MAX_BATCH = 64


def _check_image(value) -> None:
    a = np.asarray(value)
    if a.shape != (32, 32, 3):
        raise ValueError(f"expected shape (32, 32, 3), got {a.shape}")
    if a.size and (a.min() < 0 or a.max() > 255):
        raise ValueError("pixel values must be in 0..255")


class Image(BaseModel):
    pixels: list[list[list[int]]]                          # 32 x 32 x 3 RGB values in 0..255

    @field_validator("pixels")
    @classmethod
    def check_pixels(cls, v):
        _check_image(v)
        return v


class Batch(BaseModel):
    images: list[list[list[list[int]]]]                    # 1 to MAX_BATCH images of 32 x 32 x 3

    @field_validator("images")
    @classmethod
    def check_images(cls, v):
        if not 1 <= len(v) <= MAX_BATCH:
            raise ValueError(f"expected 1 to {MAX_BATCH} images, got {len(v)}")
        for image in v:
            _check_image(image)
        return v


def errors_without_input(error: ValidationError | list) -> list[dict]:
    """The errors of pydantic without the field "input" (the request may contain personal data)."""
    errors = error.errors() if isinstance(error, ValidationError) else error
    return [{"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", "")} for e in errors]


class Service:
    """The model of an artifact and its answers (shared by the two servers)."""

    def __init__(self, artifact_dir) -> None:
        self.model, self.meta = load_artifact(artifact_dir)
        self.version = self.meta["weights_sha256"][:12]

    def health(self) -> dict:
        import torch

        m = self.meta
        return {"status": "ok", "package_version": __version__,
                "model": {"arch": m["arch"], "widths": m["widths"], "classes": m["classes"],
                          "weights_sha256": m["weights_sha256"], "data_sha256": m.get("data", {}).get("sha256"),
                          "created": m.get("created")},
                "torch": torch.__version__}

    def predict(self, images: list) -> list[dict]:
        labels, proba = predict_images(self.model, self.meta, np.asarray(images, dtype=np.uint8))
        return [{"label": label, "probability": round(float(p.max()), 4)} for label, p in zip(labels, proba)]


def create_app(artifact_dir):
    """The FastAPI application of the artifact ``artifact_dir`` (its Service in ``app.state.service``)."""
    from fastapi import FastAPI, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse

    service = Service(artifact_dir)
    app = FastAPI(title="cifar-api", version=__version__)
    app.state.service = service

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": errors_without_input(exc.errors())})

    @app.get("/health")
    def health() -> dict:
        return app.state.service.health()

    @app.post("/predict")
    def predict(image: Image):
        try:
            answer = app.state.service.predict([image.pixels])[0]
        except Exception:                                  # noqa: BLE001 - logged, a clean 500 for the client
            log.exception("prediction failed")
            return JSONResponse(status_code=500, content={"detail": "internal error"})
        return {**answer, "model_version": app.state.service.version}

    @app.post("/predict_batch")
    def predict_batch(batch: Batch):
        try:
            answers = app.state.service.predict(batch.images)
        except Exception:                                  # noqa: BLE001
            log.exception("prediction failed")
            return JSONResponse(status_code=500, content={"detail": "internal error"})
        return {"predictions": answers, "model_version": app.state.service.version}

    return app


def app_from_env():
    """create_app on the folder of the variable MODEL_DIR (default "artifact"): ``uvicorn --factory``."""
    return create_app(os.environ.get("MODEL_DIR", "artifact"))


class Handler(BaseHTTPRequestHandler):
    """The same routes with the standard library; the Service is ``self.server.service``."""

    protocol_version = "HTTP/1.1"                           # keep-alive: one connection for many requests
    disable_nagle_algorithm = True                          # headers and body leave at once (no 40 ms wait)

    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(200, self.server.service.health())
        else:
            self._send(404, {"detail": "Not Found"})

    def do_POST(self) -> None:
        routes = {"/predict": Image, "/predict_batch": Batch}
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path not in routes:
            self._send(404, {"detail": "Not Found"})
            return
        try:
            request = routes[self.path].model_validate_json(body or b"{}")
        except ValidationError as error:
            self._send(422, {"detail": errors_without_input(error)})
            return
        service = self.server.service
        try:
            if self.path == "/predict":
                answer = {**service.predict([request.pixels])[0], "model_version": service.version}
            else:
                answer = {"predictions": service.predict(request.images), "model_version": service.version}
        except Exception:                                  # noqa: BLE001 - logged, a clean 500 for the client
            log.exception("prediction failed")
            self._send(500, {"detail": "internal error"})
            return
        self._send(200, answer)

    def log_message(self, format, *args) -> None:          # one DEBUG line per request, not stderr
        log.debug("%s %s", self.address_string(), format % args)


def make_server(artifact_dir, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """The fallback server (http.server), not started: ``server.serve_forever()``; port 0 = a free port
    (``server.server_address[1]``); its Service in ``server.service``."""
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.service = Service(artifact_dir)
    return server


def has_fastapi() -> bool:
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        return False
    return True


@contextlib.contextmanager
def running(artifact_dir, backend: str = "auto", host: str = "127.0.0.1"):
    """Serve the artifact in a thread, on a free port, and yield its base URL ("http://127.0.0.1:<port>"); the server
    is ALWAYS stopped and its port freed at the end of the block, even after an error. backend: "fastapi" (uvicorn),
    "stdlib" (http.server) or "auto" (FastAPI if installed)."""
    if backend == "auto":
        backend = "fastapi" if has_fastapi() else "stdlib"
    if backend == "fastapi":
        import uvicorn

        server = uvicorn.Server(uvicorn.Config(create_app(artifact_dir), host=host, port=0, log_level="warning",
                                               lifespan="off"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 30
            while not server.started:
                if not thread.is_alive() or time.monotonic() > deadline:
                    raise RuntimeError("the uvicorn server did not start")
                time.sleep(0.01)
            port = server.servers[0].sockets[0].getsockname()[1]
            yield f"http://{host}:{port}"
        finally:
            server.should_exit = True
            thread.join(10)
    elif backend == "stdlib":
        server = make_server(artifact_dir, host, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://{host}:{server.server_address[1]}"
        finally:
            server.shutdown()                              # stop serve_forever, then free the port
            server.server_close()
            thread.join(10)
    else:
        raise ValueError(f"backend must be 'auto', 'fastapi' or 'stdlib', got {backend!r}")


def main(argv: list[str] | None = None) -> int:
    """python -m cifarprod.serve [--artifact DIR] [--host HOST] [--port PORT] [--backend auto|fastapi|stdlib]."""
    parser = argparse.ArgumentParser(prog="cifarprod.serve", description="Serve the artifact of a model over HTTP.")
    parser.add_argument("--artifact", default=os.environ.get("MODEL_DIR", "artifact"),
                        help="the artifact folder (default: $MODEL_DIR, else %(default)s)")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--backend", choices=["auto", "fastapi", "stdlib"], default="auto")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    backend = args.backend if args.backend != "auto" else ("fastapi" if has_fastapi() else "stdlib")
    try:
        if backend == "fastapi":
            import uvicorn

            app = create_app(args.artifact)
            log.info("serving %s on http://%s:%d (FastAPI)", args.artifact, args.host, args.port)
            uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
        else:
            server = make_server(args.artifact, args.host, args.port)
            log.info("serving %s on http://%s:%d (http.server)", args.artifact, args.host, server.server_address[1])
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
    except (OSError, ValueError, KeyError) as err:
        log.error("cannot serve %s: %s", args.artifact, err)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
