###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from primus.tools.benchmark.rccl_runner import add_rccl_parser


def register_subcommand(subparsers):
    """
    primus-cli benchmark <suite> [suite-specific-args]
    suites: gemm | attention | rccl
    """
    parser = subparsers.add_parser("benchmark", help="Run performance benchmarks (GEMM / Attention / RCCL).")
    suite_parsers = parser.add_subparsers(dest="suite", required=True)

    # ---------- GEMM ----------
    gemm = suite_parsers.add_parser("gemm", help="GEMM microbench.")
    gemm.add_argument(
        "--model",
        type=str,
        default="all",
        help=("Run all models with --model=all, or a specific one, e.g. --model=Llama2_7B"),
    )
    gemm.add_argument(
        "--mbs-list",
        type=int,
        nargs="+",
        default=[1],
        help="Micro-batch sizes to test, e.g. --mbs-list 1 2 4",
    )
    gemm.add_argument("--output", default="benchmark_result", help="Directory to save GEMM markdown results")

    # ---------- RCCL ----------
    rccl = suite_parsers.add_parser("rccl", help="RCCL collectives bench.")
    add_rccl_parser(rccl)
    # rccl.add_argument(
    #     "--collective", choices=["all_reduce", "all_gather", "reduce_scatter", "broadcast"],
    #     default="all_reduce",
    #     help="Collective to benchmark."
    # )
    # rccl.add_argument(
    #     "--seq",
    #     type=int,
    #     default=4096,
    #     help="Sequence length for the benchmark tensor."
    # )
    # rccl.add_argument(
    #     "--hidden-size",
    #     type=int,
    #     default=8192,
    #     help="Hidden size for the benchmark tensor."
    # )
    # rccl.add_argument(
    #     "--topk",
    #     type=int,
    #     default=5,
    #     help="Top-K values to average for performance summary."
    # )
    # rccl.add_argument(
    #     "--duration",
    #     type=int,
    #     default=60,
    #     help="Benchmark duration in seconds."
    # )

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

    # if suite == "gemm":
    #     from primus.tools.benchmark.gemm.benchmark_gemm import benchmark_gemm
    #     benchmark_gemm(args.model, args.output, args.mbs_list)
    #     return
    # elif suite == "rccl":
