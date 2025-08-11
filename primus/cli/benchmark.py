###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


def register_subcommand(subparsers):
    """
    primus-cli benchmark <suite> [suite-specific-args]
    suites: gemm | attention | rccl
    """
    parser = subparsers.add_parser("benchmark", help="Run performance benchmarks (GEMM / Attention / RCCL).")
    suite_parsers = parser.add_subparsers(dest="suite", required=True)

    # ---------- GEMM ----------
    gemm = suite_parsers.add_parser("gemm", help="GEMM microbench.")
    from primus.tools.benchmark.gemm_runner import add_gemm_parser

    add_gemm_parser(gemm)

    # ---------- RCCL ----------
    # rccl = suite_parsers.add_parser("rccl", help="RCCL collectives bench.")
    # from primus.tools.benchmark.rccl_runner import add_rccl_parser
    # add_rccl_parser(rccl)

    return parser


def run(args, extra_args):
    """
    Execute the benchmark command.
    This can internally call Megatron / TorchTitan hooks, or profile.py scripts.
    """

    suite = args.suite
    print(f"[Primus:Benchmark] suite={suite} args={args}")

    from primus.tools.utils import init_distributed_if_needed as init_dist

    init_dist()

    if suite == "gemm":
        from primus.tools.benchmark.gemm_runner import run_gemm_benchmark

        run_gemm_benchmark(args)
    # elif suite == "rccl":
