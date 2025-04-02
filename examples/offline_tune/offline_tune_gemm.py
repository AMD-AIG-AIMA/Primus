import argparse
import copy
import os
import shlex
import subprocess
import time
from multiprocessing import Process, Queue


def is_hip():
    import torch

    if torch.version.hip is not None:
        return True
    return False


def worker(device_id, tune_gemm_results_file_path, task_queue):
    env = os.environ.copy()
    if is_hip():
        env["HIP_VISIBLE_DEVICES"] = device_id
    else:
        env["CUDA_VISIBLE_DEVICES"] = device_id
    env["HIPBLASLT_TUNING_FILE"] = tune_gemm_results_file_path

    while True:
        script = task_queue.get()
        if script is None:
            break
        print(f"Device {device_id} processing: {script}")
        subprocess.run(shlex.split(script), check=True, env=env)


class OfflineTuneGemm:

    def __init__(self, dump_gemm_shape_file_path):
        self.HIPBLIST_BENCH = "/opt/rocm/bin/hipblaslt-bench "
        self.ROTATING_BUFFER = 512
        self.RUN_NUMS = 20
        self.REQUESTED_SOLUTION = -1
        self.SKIP_LOW_SOLUTION = 0.7

        self.src_script_dict_list = []
        self.src_script_list = []
        self.tune_script_dict_list = []
        self.tune_script_list = []
        self.process_raw_dump(dump_gemm_shape_file_path)

    def process_raw_dump(self, dump_gemm_shape_file_path):
        with open(dump_gemm_shape_file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        lines = list(set(lines))
        lines.sort()

        for line in lines:
            line = line.strip().split(" ")
            line = [item for item in line if item.strip()]
            if line[0] == "hipblaslt-bench":
                src_script_dict = {}
                for item in line[1:]:
                    if item.startswith("--") or item.startswith("-"):
                        key = item
                    else:
                        src_script_dict[key] = item
                # src script
                src_script_dict["--rotating"] = self.ROTATING_BUFFER
                src_script_dict["--cold_iters"] = self.RUN_NUMS
                src_script_dict["--iters"] = self.RUN_NUMS
                src_script = self.HIPBLIST_BENCH + " ".join(f"{k} {v}" for k, v in src_script_dict.items())
                self.src_script_dict_list.append(src_script_dict)
                self.src_script_list.append(src_script)
                # tune script
                tune_script_dict = copy.deepcopy(src_script_dict)
                del tune_script_dict["--algo_method"]
                del tune_script_dict["--solution_index"]
                tune_script_dict["--requested_solution"] = self.REQUESTED_SOLUTION
                tune_script_dict["--skip_slow_solution_ratio"] = self.SKIP_LOW_SOLUTION
                tune_script = self.HIPBLIST_BENCH + " ".join(f"{k} {v}" for k, v in tune_script_dict.items())
                self.tune_script_dict_list.append(tune_script_dict)
                self.tune_script_list.append(tune_script)

    def tune(self, tune_gemm_results_file_path, device_ids=["0"]):
        task_queue = Queue()
        for script in self.tune_script_list:
            task_queue.put(script)
        for _ in device_ids:
            task_queue.put(None)

        start_time = time.time()
        processes = []
        for device_id in device_ids:
            p = Process(target=worker, args=(device_id, tune_gemm_results_file_path, task_queue))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(
            f"Tune cases Nums: {len(self.tune_script_list)}. Elapsed Time: {elapsed_time:.2f} s",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-shape-path", type=str)
    parser.add_argument("--tune-result-path", type=str)
    parser.add_argument("--num-devices", type=int, default=1)
    args = parser.parse_args()
    device_ids = [str(i) for i in range(args.num_devices)]

    tuner = OfflineTuneGemm(args.dump_shape_path)
    tuner.tune(args.tune_result_path, device_ids)
