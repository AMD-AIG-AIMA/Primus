###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

import argparse
import os

import torch
import torch.distributed as dist
from global_vars import LOCAL_RANK, RANK, set_hostnames
from inter_node_comm import run_inter_node_comm
from inter_node_comm_p2p import run_inter_node_comm_p2p
from intra_node_comm import run_intra_node_comm
from square_gemm import run_square_gemm
from utility import (
    gather_hostnames,
    get_first_ib_unidirectional_bandwidth,
    log,
    md_to_pdf,
    remove_file,
)


def setup():
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group("nccl")
    set_hostnames(gather_hostnames())


def cleanup():
    dist.destroy_process_group()


def main(args):
    setup()

    if RANK == 0:
        bw = get_first_ib_unidirectional_bandwidth()
        log(f"=======IB Bandwidth roofline (GB/s)=======")
        log(f"Bandwidth of first IB device of Node 0 : {bw:.2f} GB/s")
        args.ib_bw = bw

        if not os.path.isdir(args.dump_path):
            log(f"mkdir {args.dump_path}")
            os.makedirs(args.dump_path)

    args.markdown_file = f"{args.dump_path}/{args.report_file_name}.md"
    args.pdf_file = f"{args.dump_path}/{args.report_file_name}.pdf"
    remove_file(args.markdown_file)

    # run tests
    run_square_gemm(args)
    run_intra_node_comm(args)
    run_inter_node_comm(args)
    run_inter_node_comm_p2p(args)

    if RANK == 0 and args.save_pdf:
        md_to_pdf(args.markdown_file, args.pdf_file)

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-path", type=str, default="output/preflight")
    parser.add_argument("--report-file-name", type=str, default="preflight_report")
    parser.add_argument("--disable-pdf", dest="save_pdf", action="store_false")
    parser.add_argument("--disable-plot", dest="plot", action="store_false")
    args = parser.parse_args()

    main(args)
