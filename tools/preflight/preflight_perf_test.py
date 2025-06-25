###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

import argparse
import json
import os

import torch
import torch.distributed as dist
from global_vars import LOCAL_RANK, RANK, set_hostnames
from inter_node_comm import run_inter_node_comm
from inter_node_comm_p2p import run_inter_node_comm_p2p
from inter_node_ring_p2p import run_inter_node_ring_p2p
from intra_node_comm import run_intra_node_comm
from square_gemm import run_square_gemm
from tools.preflight.flash_attention import run_flash_attention
from utility import (
    gather_hostnames,
    get_first_ib_unidirectional_bandwidth,
    log,
    md_to_pdf,
    remove_file,
)


def setup(args):
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group(args.backend)
    set_hostnames(gather_hostnames())


def cleanup():
    dist.destroy_process_group()


def main(args):
    setup(args)

    if RANK == 0:
        bw = get_first_ib_unidirectional_bandwidth()
        log(f"=======IB Bandwidth roofline (GB/s)=======")
        log(f"Bandwidth of first IB device of Node 0 : {bw:.2f} GB/s")
        args.ib_bw = bw

        if not os.path.isdir(args.dump_path):
            log(f"mkdir {args.dump_path}")
            os.makedirs(args.dump_path)

    args.markdown_file = f"{args.dump_path}/{args.report_file_name}.md"
    args.json_file = f"{args.dump_path}/{args.report_file_name}.json"
    args.pdf_file = f"{args.dump_path}/{args.report_file_name}.pdf"
    remove_file(args.markdown_file)

    result_json = {}
    if args.single_node:
        print("Running Single Node Check")
        result_json["squire_gemm"] = run_square_gemm(args)
        result_json["flash_attn"] = run_flash_attention(args)
    if args.intra:
        print("Running Intra Node Check")
        result_json["intra_node_comm"] = run_intra_node_comm(args)
    if args.multi_node:
        print("Running Multi Node Check")
        result_json["inter_node_comm"] = run_inter_node_comm(args)
        result_json["inter_node_comm_p2p"] = run_inter_node_comm_p2p(args)
        result_json["inter_node_ring_p2p"] = run_inter_node_ring_p2p(args)
    if not args.single_node and not args.multi_node:
        print("FATAL!No Check executed")
    # run tests

    if RANK == 0 and args.save_pdf:
        md_to_pdf(args.markdown_file, args.pdf_file)
    if RANK == 0 and args.save_json:
        json.dump(result_json, open(args.json_file, "w"))
    cleanup()


if __name__ == "__main__":
    print(os.environ)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-path", type=str, default="output/preflight")
    parser.add_argument("--report-file-name", type=str, default="preflight_report")
    parser.add_argument("--disable-pdf", dest="save_pdf", action="store_false")
    parser.add_argument("--disable-plot", dest="plot", action="store_false")
    parser.add_argument("--disable-json", dest="save_json", action="store_false")
    parser.add_argument('--single_node', action='store_true', help='Run single node check')
    parser.add_argument("--intra",action='store_true')
    parser.add_argument('--multi_node', action='store_true', help='Run multi node check')
    parser.add_argument('--all', action='store_true',
                        help='Run all checks (equivalent to --single_node and --multi_node)')
    parser.add_argument("--backend", default="nccl")
    args = parser.parse_args()


    def env_flag(name, default):
        val = os.getenv(name)
        if val is None:
            return default
        return val.lower() in ["1", "true", "yes", "on"]

    print('LOAD ENV')
    args.dump_path = os.getenv("DUMP_PATH", args.dump_path)
    args.report_file_name = os.getenv("REPORT_FILE_NAME", args.report_file_name)
    args.backend = os.getenv("BACKEND", args.backend)
    args.save_pdf = env_flag("SAVE_PDF", args.save_pdf)
    args.save_json = env_flag("SAVE_JSON", args.save_json)
    args.plot = env_flag("PLOT", args.plot)
    args.single_node = env_flag("SINGLE_NODE", args.single_node)
    args.multi_node = env_flag("MULTI_NODE", args.multi_node)
    args.intra = env_flag("INTRA",args.intra)
    args.all = env_flag("ALL", args.all)
    print(args)
    if args.all:
        args.single_node = True
        args.multi_node = True
        args.intra = True
    main(args)
