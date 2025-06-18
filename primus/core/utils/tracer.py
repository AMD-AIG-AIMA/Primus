import argparse
import os
import json
import traceback
from dataclasses import dataclass

from opentelemetry import trace
from opentelemetry.trace import Tracer, SpanKind, Status, StatusCode
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
    ConsoleSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


_tracer = None

def get_tracer():
    """Get the global TrainTracer instance, initializing it if necessary."""
    global _tracer
    if _tracer is None:
        init_tracer()
    return _tracer

def init_tracer():
    global _tracer
    if _tracer is None:
        config = TrainTracerConfig.from_env()
        _tracer = TrainTracer(config)
    return _tracer


@dataclass
class TrainTracerConfig:
    rank: int = 0
    world_size: int = 1
    enabled: bool = False
    file_path: str = "logs/traces.json"
    collector_endpoint: str = "http://localhost:4318/v1/traces"
    tracer_name: str = "train_tracer"
    debug: bool = False

    @staticmethod
    def from_env() -> "TrainTracerConfig":
        return TrainTracerConfig(
            rank=int(os.getenv("RANK", 0)),
            world_size=int(os.getenv("WORLD_SIZE", 1)),
            enabled=os.getenv("TRACER_ENABLED", "true").lower() == "true",
            file_path=os.getenv("TRACER_FILE_PATH", "logs/traces.json"),
            collector_endpoint=os.getenv("TRACER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces"),
            tracer_name=os.getenv("TRACER_NAME", "train_tracer"),
            debug=os.getenv("TRACER_DEBUG", "false").lower() == "true"
        )

    @staticmethod
    def from_args(args=None) -> "TrainTracerConfig":
        parser = argparse.ArgumentParser()
        parser.add_argument("--rank", type=int, default=0)
        parser.add_argument("--world-size", type=int, default=1)
        parser.add_argument("--tracer-enabled", action="store_true")
        parser.add_argument("--tracer-disabled", dest="tracer_enabled", action="store_false")
        parser.set_defaults(tracer_enabled=True)
        parser.add_argument("--tracer-file-path", type=str, default="logs/traces.json")
        parser.add_argument("--tracer-otlp-endpoint", type=str, default="http://localhost:4318/v1/traces")
        parser.add_argument("--tracer-name", type=str, default="train_tracer")
        parser.add_argument("--tracer-debug", action="store_true")

        parsed = parser.parse_args(args)
        return TrainTracerConfig(
            rank=parsed.tracer_rank,
            world_size=parsed.tracer_world_size,
            enabled=parsed.tracer_enabled,
            file_path=parsed.tracer_file_path,
            collector_endpoint=parsed.tracer_otlp_endpoint,
            tracer_name=parsed.tracer_name,
            debug=parsed.tracer_debug
        )


class FileSpanExporter(SpanExporter):
    def __init__(self, filepath):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def export(self, spans):
        with open(self.filepath, "a") as f:
            for span in spans:
                f.write(json.dumps(span.to_json(), ensure_ascii=False) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


class TrainTracer:
    def __init__(self, config: TrainTracerConfig):
        self.enabled = config.enabled
        if not self.enabled:
            return

        resource = Resource.create({
            "service.name": config.tracer_name,
            "rank": config.rank,
            "world_size": config.world_size
        })

        self.provider = TracerProvider(resource=resource)
        self.tracer = trace.get_tracer(config.tracer_name)

        if config.file_path:
            file_exporter = FileSpanExporter(config.file_path)
            self.provider.add_span_processor(BatchSpanProcessor(file_exporter))

        if config.collector_endpoint:
            otlp_exporter = OTLPSpanExporter(endpoint=config.collector_endpoint)
            self.provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        if config.debug:
            console_exporter = BatchSpanProcessor(ConsoleSpanExporter())
            self.provider.add_span_processor(console_exporter)
        trace.set_tracer_provider(self.provider)

        self.training_span = None
        self.iter_spans = {}

    def start_training(self, model: str):
        if not self.enabled:
            return
        self.training_span = self.tracer.start_span(
            "training_run", kind=SpanKind.INTERNAL,
            attributes={
                "model": model,
            }
        )
        self.training_span.__enter__()

    def end_training(self, success=True):
        if not self.enabled:
            return
        if not self.training_span:
            return
        if not success:
            self.training_span.set_status(Status(StatusCode.ERROR, "Training failed"))
        self.training_span.__exit__(None, None, None)

    def start_iter(self, iter_id: int, epoch: int):
        if not self.enabled:
            return
        span = self.tracer.start_span(
            name=f"iteration_{iter_id}",
            kind=SpanKind.INTERNAL,
            attributes={
                "iter_id": iter_id,
                "epoch": epoch
            }
        )
        span.__enter__()
        self.iter_spans[iter_id] = span

    def end_iter(self, iter_id: int, **attributes):
        if not self.enabled:
            return
        span = self.iter_spans.pop(iter_id, None)
        if span:
            for k, v in attributes.items():
                span.set_attribute(k, v)
            span.__exit__(None, None, None)

    def record_iter_event(self, iter_id: int, name: str, duration_ms: float):
        if not self.enabled:
            return
        span = self.iter_spans.get(iter_id)
        if span:
            span.add_event(name=name, attributes={"duration_ms": duration_ms})

    def record_checkpoint(self, path: str, epoch: int, duration_ms: float, size_mb: float, success: bool):
        if not self.enabled:
            return
        with self.tracer.start_as_current_span("checkpoint_save") as span:
            span.set_attributes({
                "checkpoint_path": path,
                "epoch": epoch,
                "duration_ms": duration_ms,
                "checkpoint_size_mb": size_mb,
                "success": success
            })

    def record_error(self, node_rank: int, gpu_id: int, error_type: str, err: Exception, fatal=True):
        if not self.enabled:
            return
        with self.tracer.start_as_current_span("training_error") as span:
            span.set_status(Status(StatusCode.ERROR, str(err)))
            span.set_attributes({
                "node_rank": node_rank,
                "gpu_id": gpu_id,
                "error_type": error_type,
                "fatal": fatal
            })
            span.add_event("exception", attributes={
                "exception.type": type(err).__name__,
                "exception.message": str(err),
                "exception.stacktrace": traceback.format_exc()
            })