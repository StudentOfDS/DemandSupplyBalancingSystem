from __future__ import annotations

import importlib
import importlib.util
import os

from nexus_engine.api.app import _request_from_dict, _response_to_dict, plan


if importlib.util.find_spec("celery") is not None:
    celery_module = importlib.import_module("celery")
    Celery = celery_module.Celery
    celery_app = Celery(
        "nexus_engine",
        broker=os.getenv("REDIS_URL", "redis://redis:6379/0"),
        backend=os.getenv("REDIS_URL", "redis://redis:6379/0"),
    )

    @celery_app.task(name="nexus_engine.plan")
    def run_plan(payload: dict[str, object]) -> dict[str, object]:
        return run_plan_sync(payload)
else:
    class LocalCeleryApp:
        def task(self, name: str | None = None):
            def decorator(func):
                func.task_name = name or func.__name__
                return func

            return decorator

    celery_app = LocalCeleryApp()

    @celery_app.task(name="nexus_engine.plan")
    def run_plan(payload: dict[str, object]) -> dict[str, object]:
        return run_plan_sync(payload)


def run_plan_sync(payload: dict[str, object]) -> dict[str, object]:
    request = _request_from_dict(payload)
    return _response_to_dict(plan(request))
