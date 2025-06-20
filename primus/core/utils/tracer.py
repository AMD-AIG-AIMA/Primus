import argparse
import os
import json
import torch
import traceback
from dataclasses import dataclass

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from opentelemetry.trace import Tracer, SpanKind, Status, StatusCode, SpanContext, TraceFlags, set_span_in_context, \
    NonRecordingSpan
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
    ConsoleSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.id_generator import IdGenerator
import uuid
import os

def init_tracer():
    config = TrainTracerConfig.from_env()
    return  TrainTracer(config)



class RankIdGenerator(IdGenerator):
    def __init__(self, rank: int):
        self.rank = rank
        self.prefix = (uuid.uuid4().int >> 64) ^ (os.getpid() << 8) ^ rank

    def generate_trace_id(self) -> int:
        # 仍然只由 rank 0 生成
        return uuid.uuid4().int & ((1 << 128) - 1)

    def generate_span_id(self) -> int:
        # span_id 是 64bit，低 48bit 随机，高 16bit 加个 rank 做区分
        random_part = uuid.uuid4().int & 0x0000FFFFFFFFFFFF
        span_id = ((self.rank & 0xFFFF) << 48) | random_part
        return span_id

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

        self.rank = config.rank
        self.world_size = config.world_size
        self.iter_spans = {}
        self.training_span = None
        self.rank_span = None
        print(f'Init tracer for rank {self.rank}')
        resource = Resource.create({
            "service.name": f'{config.tracer_name}',
            "rank": self.rank,
            "world_size": self.world_size
        })

        self.provider = TracerProvider(
            resource=resource,
            sampler=ALWAYS_ON,
            id_generator=RankIdGenerator(rank=config.rank)
        )
        self.tracer = self.provider.get_tracer(f"rank_{self.rank}")

        if config.file_path:
            self.provider.add_span_processor(BatchSpanProcessor(FileSpanExporter(config.file_path)))
        if config.collector_endpoint:
            self.provider.add_span_processor(BatchSpanProcessor(
                OTLPSpanExporter(endpoint=config.collector_endpoint)))
        if config.debug:
            self.provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    def start_training(self, model: str):
        if not self.enabled:
            return

        self.rank_span = self.tracer.start_span(
            name=f"rank_{self.rank}",
            context=Context(),
            kind=SpanKind.INTERNAL,
            attributes={"rank": self.rank}
        )
        print(f"[Tracer Rank {self.rank}] rank trace_id {self.rank_span.get_span_context().trace_id} span_id {self.rank_span.get_span_context().span_id}")
        self.rank_span.__enter__()

    def end_training(self, success=True):
        if not self.enabled:
            return
        if self.rank_span:
            self.rank_span.__exit__(None, None, None)
        if self.training_span:
            if not success:
                self.training_span.set_status(Status(StatusCode.ERROR, "Training failed"))
            self.training_span.__exit__(None, None, None)

    def start_iter(self, iter_id: int):
        if not self.enabled:
            return
        print(f"tracer start iter {iter_id}")
        ctx = set_span_in_context(self.rank_span)
        span = self.tracer.start_span(
            name=f"iteration_{iter_id}",
            context=ctx,
            kind=SpanKind.INTERNAL,
            attributes={"iter_id": iter_id}
        )
        trace_id = span.get_span_context().trace_id
        span_id = span.get_span_context().span_id
        print(f"[Tracer Rank {self.rank}] iter {iter_id} trace_id {trace_id} span_id = {span_id:x}")
        span.__enter__()
        self.iter_spans[iter_id] = span

    def end_iter(self, iter_id: int, attributes: dict):
        if not self.enabled:
            return
        print(f"tracer end iter {iter_id}. attributes={attributes}")
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
        ctx = set_span_in_context(self.rank_span)
        with self.tracer.start_as_current_span("checkpoint_save", context=ctx) as span:
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
        ctx = set_span_in_context(self.rank_span)
        with self.tracer.start_as_current_span("training_error", context=ctx) as span:
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