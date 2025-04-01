import argparse
import copy
import os
import shlex
import subprocess


def is_hip():
    import torch

    if torch.version.hip is not None:
        return True
    return False


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

    # TODO: use more device to tune
    def tune(self, tune_gemm_results_file_path, device_id="0"):
        env = os.environ.copy()
        if is_hip():
            env.update({"HIP_VISIBLE_DEVICES": device_id})
        else:
            env.update({"CUDA_VISIBLE_DEVICES": device_id})
        env.update({"HIPBLASLT_TUNING_FILE": tune_gemm_results_file_path})

        for idx, script in enumerate(self.tune_script_list):
            print(f"Tune[{idx}/{len(self.tune_script_list)}]:{script}")
            subprocess.run(shlex.split(script), env=env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-shape-pathh", type=str)
    parser.add_argument("--tune-result-path", type=str)
    # parser.add_argument("--device-id", type=str, default="0")
    args = parser.parse_args()

    tuner = OfflineTuneGemm(args.dump_shape_path)
    tuner.tune(args.tune_result_path)
