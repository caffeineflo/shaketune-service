"""Manual span helpers for operations auto-instrumentation cannot cover.

SDK initialization, exporters, and resource attributes are handled by
the `opentelemetry-instrument` CLI wrapper and OTEL_* env vars.
"""

from __future__ import annotations

import asyncio
import functools
from contextlib import asynccontextmanager
from typing import Any

from opentelemetry import trace

_TRACER_NAME = "shaketune-service"

tracer = trace.get_tracer(_TRACER_NAME)


def span(operation: str, *, component: str = "", attributes: dict[str, Any] | None = None):
    """Decorator for functions that should be traced."""
    span_name = f"{component}.{operation}" if component else operation

    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                with tracer.start_as_current_span(span_name, attributes=attributes or {}):
                    return await func(*args, **kwargs)
            return wrapper
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with tracer.start_as_current_span(span_name, attributes=attributes or {}):
                    return func(*args, **kwargs)
            return wrapper
    return decorator


@asynccontextmanager
async def trace_operation(operation: str, component: str = "", **attrs):
    """Context manager for tracing a block of code."""
    span_name = f"{component}.{operation}" if component else operation
    with tracer.start_as_current_span(span_name, attributes=attrs) as s:
        yield s
