"""Prometheus text for the CUDA server's GET /metrics: counters of finished requests, gauges read at scrape time."""
# Ported from TensorFold v0.6.0, commit 4447ac3 "Prometheus /metrics on both servers (#110)" (Apache-2.0): its CUDA part.

from __future__ import annotations

import threading
from typing import Any

PREFIX = "tensorfold:"             # upstream's names
HEALTH = "tensorfold_health:"      # this server's /health fields, named as /health names them
# Request and time-to-first-token histograms share these upper edges. +Inf is added when rendered.
BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)
_MADE = threading.Lock()
_TRIES = 3                         # a decoder's tables read while its thread changes them: read again


class Histogram:
    """Counts in one bucket each. Render adds them up into Prometheus's cumulative buckets."""

    def __init__(self) -> None:
        self.counts = [0] * (len(BUCKETS) + 1)
        self.total = 0.0
        self.n = 0

    def observe(self, value: float) -> None:
        value = max(0.0, float(value))
        self.n += 1
        self.total += value
        for i, edge in enumerate(BUCKETS):
            if value <= edge:
                self.counts[i] += 1
                return
        self.counts[-1] += 1

    def copy(self) -> "Histogram":
        other = Histogram()
        other.counts = list(self.counts)
        other.total, other.n = self.total, self.n
        return other


class Metrics:
    """Counters and histograms of finished requests. Gauges are not stored here."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.prompt = 0
        self.generation = 0
        self.drafted = 0
        self.accepted = 0
        self.latency = Histogram()
        self.ttft = Histogram()

    def add(self, *, prompt: int, generation: int, drafted: int, accepted: int,
            latency: float | None, ttft: float | None) -> None:
        with self.lock:
            self.prompt += int(prompt)
            self.generation += int(generation)
            self.drafted += int(drafted)
            self.accepted += int(accepted)
            if latency is not None:
                self.latency.observe(latency)
            if ttft is not None:
                self.ttft.observe(ttft)


def of(app: Any) -> Metrics:
    """The app's counters, made on first use."""

    with _MADE:
        found = app.__dict__.get("metrics")
        if found is None:
            found = app.__dict__["metrics"] = Metrics()
        return found


def note(app: Any, *, prompt: int = 0, generation: int = 0, drafted: int = 0, accepted: int = 0,
         latency: float | None = None, ttft: float | None = None) -> None:
    """Fold one finished request. A missing app is a no-op."""

    if app is None:
        return
    of(app).add(prompt=prompt, generation=generation, drafted=drafted, accepted=accepted,
                latency=latency, ttft=ttft)


def render(app: Any) -> str:
    """The scrape body, ending in a newline: host-side reads only, no lock the engine's rounds take."""

    metrics = of(app)
    with metrics.lock:
        prompt, generation = metrics.prompt, metrics.generation
        drafted, accepted = metrics.drafted, metrics.accepted
        latency, ttft = metrics.latency.copy(), metrics.ttft.copy()
    running, waiting = _read(lambda: _requests(app), (0, 0))
    lines: list[str] = []
    _family(lines, "requests_running", "gauge", "Requests in prefill or decode.",
            [f"{PREFIX}requests_running {running}"])
    _family(lines, "requests_waiting", "gauge", "Requests queued or held until a lane is free.",
            [f"{PREFIX}requests_waiting {waiting}"])
    _family(lines, "prompt_tokens_total", "counter", "Prompt tokens of finished requests.",
            [f"{PREFIX}prompt_tokens_total {prompt}"])
    _family(lines, "generation_tokens_total", "counter", "Generated tokens of finished requests.",
            [f"{PREFIX}generation_tokens_total {generation}"])
    _family(lines, "kv_cache_usage_ratio", "gauge",
            "Tokens in a stream cache divided by that stream's context window.",
            [f'{PREFIX}kv_cache_usage_ratio{{pool="{pool}"}} {_num(ratio)}'
             for pool, ratio in _read(lambda: _pools(app), [("0", 0.0)])])
    _family(lines, "mtp_drafted_total", "counter", "Draft tokens verified on finished requests.",
            [f"{PREFIX}mtp_drafted_total {drafted}"])
    _family(lines, "mtp_accepted_total", "counter", "Draft tokens kept on finished requests.",
            [f"{PREFIX}mtp_accepted_total {accepted}"])
    _histogram(lines, "request_latency_seconds", "Seconds from arrival to the reply leaving.", latency)
    _histogram(lines, "time_to_first_token_seconds", "Seconds from arrival to the first generated token.", ttft)
    _health(lines, app)
    return "\n".join(lines) + "\n"


def send(handler: Any, app: Any) -> None:
    """Write ``render`` as Prometheus text, version 0.0.4."""

    body = render(app).encode()
    try:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
    except (BrokenPipeError, ConnectionResetError):
        handler.close_connection = True


# /health's fields under HEALTH: (field, type, help); the totals are the finished requests' engine stats
_HEALTH_TOTALS = (
    ("requests_total", "counter", "Finished requests (as /health's requests_total)."),
    ("completion_tokens_total", "counter", "Reply tokens, the running replies' tokens so far included."),
    ("cached_tokens_total", "counter", "Prompt tokens found cached, finished requests."),
    ("rounds_total", "counter", "Decode rounds of finished requests."),
    ("prefill_seconds_total", "counter", "Engine prefill seconds of finished requests."),
    ("decode_seconds_total", "counter", "Engine decode seconds of finished requests."),
)
_HEALTH_GAUGES = (
    ("context_length", "Tokens a request's prompt and reply may hold (the server's effective window)."),
    ("streams_max", "Streams the concurrent engine decodes at most (--parallel)."),
    ("pool_tokens", "Token rows in the concurrent engine's shared cache pool."),
    ("pool_free_tokens", "Pool rows no stream or kept prompt holds."),
    ("kept_prompts", "Prompt states kept in the pool for reuse."),
)


def _health(lines: list[str], app: Any) -> None:
    """/health's totals and the concurrent engine's streams and pool (the same reads /health makes)."""

    from tensorfold.cuda import health

    body = _read(lambda: health.of(app).snapshot(app), None)
    if body is None:
        return
    for name, kind, help_text in _HEALTH_TOTALS:
        if isinstance(body.get(name), (int, float)):
            _family(lines, name, kind, help_text, [f"{HEALTH}{name} {_num(body[name])}"], HEALTH)
    streams = dict(body.get("streams") or {})
    if "max" in streams:
        body["streams_max"] = streams.pop("max")
    prefilling = streams.pop("prefilling", None)        # GLM's decoder counts the same streams as "filling"
    if prefilling is not None:
        streams.setdefault("filling", prefilling)
    if streams:
        _family(lines, "streams", "gauge", "The concurrent engine's streams by state.",
                [f'{HEALTH}streams{{state="{state}"}} {_num(n)}' for state, n in sorted(streams.items())
                 if isinstance(n, (int, float))], HEALTH)
    for name, help_text in _HEALTH_GAUGES:
        if isinstance(body.get(name), (int, float)):
            _family(lines, name, "gauge", help_text, [f"{HEALTH}{name} {_num(body[name])}"], HEALTH)


def _read(read, default):
    """``read()``, again if a table changed size under it; ``default`` if it keeps failing."""

    for _ in range(_TRIES):
        try:
            return read()
        except RuntimeError:              # "dictionary changed size during iteration" and the like
            continue
    return default


def _family(lines: list[str], name: str, kind: str, help_text: str, samples: list[str],
            prefix: str = PREFIX) -> None:
    full = prefix + name
    lines.append(f"# HELP {full} {help_text}")
    lines.append(f"# TYPE {full} {kind}")
    lines.extend(samples)


def _histogram(lines: list[str], name: str, help_text: str, hist: Histogram) -> None:
    full = PREFIX + name
    lines.append(f"# HELP {full} {help_text}")
    lines.append(f"# TYPE {full} histogram")
    cumulative = 0
    for edge, count in zip(BUCKETS, hist.counts):
        cumulative += count
        lines.append(f'{full}_bucket{{le="{_edge(edge)}"}} {cumulative}')
    lines.append(f'{full}_bucket{{le="+Inf"}} {cumulative + hist.counts[-1]}')
    lines.append(f"{full}_sum {_num(hist.total)}")
    lines.append(f"{full}_count {hist.n}")


def _decoder(app: Any) -> tuple[Any, Any]:
    sched = getattr(getattr(app, "engine", None), "scheduler", None)
    return sched, getattr(sched, "decoder", None)


def _requests(app: Any) -> tuple[int, int]:
    """(running, waiting): a concurrent engine's lanes and queue, else /health's live requests and the turn queue."""

    sched, decoder = _decoder(app)
    if decoder is not None and hasattr(decoder, "live"):
        live = decoder.live
        running = int(live() if callable(live) else live)
        waiting = 0
        queue = getattr(sched, "waiting", None)
        if queue is not None and hasattr(queue, "qsize"):
            waiting += int(queue.qsize())
        if getattr(sched, "held", None) is not None:
            waiting += 1
        return running, waiting
    health = getattr(app, "health", None)
    running = len(getattr(health, "live", ()) or ())
    turns = getattr(app, "turns", None)
    if turns is None:
        return running, 0
    parked = getattr(turns, "parked", None)
    return running, int(parked if parked is not None else getattr(turns, "waiting", 0) or 0)


def _pools(app: Any) -> list[tuple[str, float]]:
    """One ratio per live stream. An idle server still publishes pool 0 at 0."""

    window = _window(app)
    lengths = _lengths(app)
    if not lengths:
        return [("0", 0.0)]
    if window <= 0:
        return [(str(i), 0.0) for i in range(len(lengths))]
    return [(str(i), min(1.0, n / window)) for i, n in enumerate(lengths)]


def _window(app: Any) -> int:
    engine = getattr(app, "engine", None)
    _, decoder = _decoder(app)
    # the effective window first: GLM's engine keeps its own as ``limit``, below the checkpoint's native window
    for owner, name in ((app, "effective_context_window"), (engine, "context_window"), (app, "context_window"),
                        (decoder, "context")):
        n = _positive(getattr(owner, name, None) if owner is not None else None)
        if n:
            return n
    return 0


def _lengths(app: Any) -> list[int]:
    _, decoder = _decoder(app)
    if decoder is None:
        return []
    streams = list(getattr(decoder, "streams", {}).values())
    streams += list(getattr(decoder, "filling", ()) or ())
    return [_occupied(stream) for stream in streams]


def _occupied(stream: Any) -> int:
    """A stream's prompt and reply so far (a ``Stream``'s ``context`` holds its reply, or its prompt and reply)."""

    context = len(getattr(stream, "context", None) or ())
    return max(context, len(getattr(stream, "prompt", None) or ()) + len(getattr(stream, "out", None) or ()))


def _positive(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def _num(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _edge(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


__all__ = ["BUCKETS", "HEALTH", "PREFIX", "Histogram", "Metrics", "note", "of", "render", "send"]
