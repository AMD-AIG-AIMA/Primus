import re
import subprocess

import torch


class FlashAttnProfiler:
    def __init__(
        self,
        batch_size,
        seq_len,
        num_head_q,
        num_head_kv,
        head_dim_qk,
        head_dim_v,
        causal,
        dtype=torch.bfloat16,
        device="cuda:0",
    ):
        #
        self.causal = causal

        #
        self.q = torch.randn(
            (batch_size, seq_len, num_head_q, head_dim_qk),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        self.k = torch.randn(
            (batch_size, seq_len, num_head_kv, head_dim_v),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        self.v = torch.randn(
            (batch_size, seq_len, num_head_kv, head_dim_v),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        self.o = torch.randn(
            (batch_size, seq_len, num_head_q, head_dim_v),
            dtype=dtype,
            device=device,
        )
        self.o_grad = torch.randn_like(self.o)

        self.tflop_fwd = 2 * batch_size * seq_len * seq_len * num_head_q * (head_dim_qk + head_dim_v) / 1e12
        if causal is True:
            self.tflop_fwd = self.tflop_fwd * 0.5
        self.tflop_bwd = self.tflop_fwd * 2.5

        # Cuda Event
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.num_runs = 100

    def profile(self):
        from flash_attn import flash_attn_func

        # warm up
        for _ in range(10):
            self.q.grad = None
            self.k.grad = None
            self.v.grad = None
            self.o = flash_attn_func(
                self.q,
                self.k,
                self.v,
                causal=self.causal,
            )
            self.o.backward(self.o_grad)
        torch.cuda.synchronize()
        # FWD
        self.start_event.record()
        for _ in range(self.num_runs):
            self.q.grad = None
            self.k.grad = None
            self.v.grad = None
            self.o = flash_attn_func(
                self.q,
                self.k,
                self.v,
                causal=self.causal,
            )
        self.end_event.record()
        torch.cuda.synchronize()
        fwd_time = self.start_event.elapsed_time(self.end_event) / self.num_runs / 1000
        fwd_tflops = self.tflop_fwd / fwd_time

        # FWD + BWD
        self.start_event.record()
        for _ in range(self.num_runs):
            self.q.grad = None
            self.k.grad = None
            self.v.grad = None
            self.o = flash_attn_func(
                self.q,
                self.k,
                self.v,
                causal=self.causal,
            )
            self.o.backward(self.o_grad)
        self.end_event.record()
        torch.cuda.synchronize()
        fwd_bwd_time = self.start_event.elapsed_time(self.end_event) / self.num_runs / 1000
        bwd_time = fwd_bwd_time - fwd_time
        bwd_tflops = self.tflop_bwd / bwd_time
        return fwd_tflops, fwd_time, bwd_tflops, bwd_time


def flash_attention_profile(
    batch_size,
    seq_len,
    num_head_q,
    num_head_kv,
    head_dim_qk,
    head_dim_v,
    causal,
    dtype=torch.bfloat16,
):
    profiler = FlashAttnProfiler(
        batch_size=batch_size,
        seq_len=seq_len,
        num_head_q=num_head_q,
        num_head_kv=num_head_kv,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        causal=causal,
        dtype=dtype,
    )
    try:
        fwd_tflops, fwd_time, bwd_tflops, bwd_time = profiler.profile()
    except RuntimeError:
        return 0, 0, 0, 0
    return fwd_tflops, fwd_time, bwd_tflops, bwd_time


class CKProfiler:
    def __init__(
        self,
        batch_size,
        seq_len,
        num_head_q,
        num_head_kv,
        head_dim_qk,
        head_dim_v,
        causal,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_head_q = num_head_q
        self.num_head_kv = num_head_kv
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_v
        self.causal = causal

        self.tflop_fwd = 2 * batch_size * seq_len * seq_len * num_head_q * (head_dim_qk + head_dim_v) / 1e12
        if causal is True:
            self.tflop_fwd = self.tflop_fwd * 0.5
        self.tflop_bwd = self.tflop_fwd * 2.5

        # Script
        self.fwd_script = [
            "tile_example_fmha_fwd",
            f"-mode={0}",
            f"-b={self.batch_size}",
            f"-h={self.num_head_q}",
            f"-h_k={self.num_head_kv}",
            f"-s={self.seq_len}",
            f"-s_k={self.seq_len}",
            f"-d={self.head_dim_qk}",
            f"-d_v={self.head_dim_v}",
            f"-prec=bf16",
            "-mask={}".format(2 if self.causal else 0),
            "-warmup=100",
            "-repeat=100",
            "-v=0",
        ]

        self.bwd_script = self.fwd_script.copy()
        self.bwd_script[0] = "tile_example_fmha_bwd"
        self.fwd_script.append(f"-lse={1}")

    def profile(self):
        fwd_time = 0
        fwd_tflops = 0
        bwd_time = 0
        bwd_tflops = 0
        # FWD
        fwd_res = subprocess.run(self.fwd_script, capture_output=True, text=True)
        fwd_match = re.search(r"([\d\.]+)\s*ms", fwd_res.stdout)
        if fwd_match:
            fwd_time = float(fwd_match.group(1)) / 1000
            fwd_tflops = self.tflop_fwd / fwd_time
        # BWD
        bwd_res = subprocess.run(self.bwd_script, capture_output=True, text=True)
        bwd_match = re.search(r"([\d\.]+)\s*ms", bwd_res.stdout)
        if bwd_match:
            bwd_time = float(bwd_match.group(1)) / 1000
            bwd_tflops = self.tflop_bwd / bwd_time

        return fwd_tflops, fwd_time, bwd_tflops, bwd_time


def ck_attention_profile(
    batch_size,
    seq_len,
    num_head_q,
    num_head_kv,
    head_dim_qk,
    head_dim_v,
    causal,
):
    profiler = CKProfiler(
        batch_size=batch_size,
        seq_len=seq_len,
        num_head_q=num_head_q,
        num_head_kv=num_head_kv,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        causal=causal,
    )
    try:
        fwd_tflops, fwd_time, bwd_tflops, bwd_time = profiler.profile()
    except RuntimeError:
        return 0, 0, 0, 0
    return fwd_tflops, fwd_time, bwd_tflops, bwd_time
