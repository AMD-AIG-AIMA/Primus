from alignorch.core.task.autotask import AutoTask
from alignorch.core.task.submitter import TaskSubmitter
from alignorch.core.utils import logger

from .polaris_client import PolarisClient


class PolarisDirectTaskSubmitter(TaskSubmitter):
    submit_method = "polaris-direct"

    polaris_current_node_resource_limit = {
        "cpu": 167,
        "gpu": 8,
        "memory": 1975,
        "shareMemory": 987,
        "ephemeralStorage": 354,
    }

    def __init__(self, auth_info: dict):
        self.client = PolarisClient(
            endpoint=auth_info.get("endpoint"),
            cluster=auth_info.get("cluster"),
            user_xcs_id=auth_info.get("user_xcs_id"),
            user_xcs_password=auth_info.get("user_xcs_password"),
        )
        logger.debug(f"client init with {auth_info}")
        self.workspace = auth_info.get("workspace", "prod")
        super().__init__()

    def connect(self):
        try:
            self.client.login()
        except Exception as e:
            logger.error(f"Connect failed due to {e}")

    def submit(self, task: AutoTask, script_arguments: dict):
        self.connect()
        workload_id = None
        if not self.is_compatible(task):
            return workload_id
        request_args = {
            "workspace": self.workspace,
            "priority": task.priority,
            "jobname": task.task_name,
            "description": f"This is an auto task submitted by AlignOrch. The arguments given: {script_arguments}",
            "script": task.fill_script(script_arguments),
            "image": task.image,
            "resource_dict": {
                "cpu": min(task.cpu_per_node, self.polaris_current_node_resource_limit["cpu"]),
                "gpu": min(max(task.gpu_per_node, 1), self.polaris_current_node_resource_limit["gpu"]),
                "memory": f"{self.polaris_current_node_resource_limit['memory']}Gi",
                "gpu_type": "nvidia.com/gpu",
                "share_memory": f"{self.polaris_current_node_resource_limit['shareMemory']}Gi",
                "ephemeral_storage": f"{self.polaris_current_node_resource_limit['ephemeralStorage']}Gi",
            },
            "worker_num": task.n_nodes,
        }
        logger.debug(f"Attempt to submit task {task.task_name} with args {request_args}")
        workload_id = self.client.create_workload(**request_args)
        return workload_id
