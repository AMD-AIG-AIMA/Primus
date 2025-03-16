import json
import re
import urllib.parse
from abc import ABC, abstractmethod

import requests


class PolarisClientHTTPException(Exception):
    def __init__(self, error_message: dict):
        self.error_message = error_message

    def __str__(self):
        return (
            f"PolarisClientHTTPException raised when calling {self.error_message.get('url')}: "
            f"({self.error_message.get('type')} # {self.error_message.get('status_code')}) "
            f"{self.error_message.get('reason')}"
        )


class PolarisRequest(ABC):
    method: str

    @abstractmethod
    def make_request_url(self) -> str:
        pass

    def make_request_headers(self) -> dict:
        return None

    def make_request_payload(self) -> dict:
        return None

    def make_request_params(self) -> dict:
        return None

    def make_request_data(self) -> dict:
        return None

    def send(self):
        request_params = {
            "method": self.method,
            "url": self.make_request_url(),
            "headers": self.make_request_headers(),
            "json": self.make_request_payload(),
            "params": self.make_request_params(),
            "data": self.make_request_data(),
        }
        # Remove None values from the dictionary
        request_params = {k: v for k, v in request_params.items() if v is not None}
        response = requests.request(**request_params)
        return self.check_http_status(response)

    @staticmethod
    def check_http_status(response: requests.Response):
        """
        Check the HTTP status code of the response.
        If the status code is in the range of 400 to 600, raise a PolarisClientHTTPException.
            - 400 to 500: client error
            - 500 to 600: server error
        """
        decode_reason = lambda text: (
            text.decode("utf-8")
            if isinstance(text, bytes)
            else text.decode("iso-8859-1") if isinstance(text, bytes) else text
        )

        error_message = None
        if 400 <= response.status_code < 500:
            error_message = {
                "status_code": response.status_code,
                "url": response.url,
                "type": "client_error",
                "reason": decode_reason(response.text),
            }
        elif 500 <= response.status_code < 600:
            error_message = {
                "status_code": response.status_code,
                "url": response.url,
                "type": "server_error",
                "reason": decode_reason(response.text),
            }
        else:
            pass

        if error_message is not None:
            raise PolarisClientHTTPException(error_message)
        else:
            return response.json()


class PolarisLoginRequest(PolarisRequest):
    method: str = "POST"

    def __init__(self, user_xcs_id: str, user_xcs_password: str, endpoint: str):
        self.user_xcs_id = user_xcs_id
        self.user_xcs_password = user_xcs_password
        self.endpoint = endpoint

    def make_request_url(self) -> str:
        return f"{self.endpoint}/api/v1/login"

    def make_request_payload(self) -> dict:
        return {"userId": self.user_xcs_id, "password": self.user_xcs_password}


class PolarisCreateWorkloadRequest(PolarisRequest):
    method: str = "POST"

    def __init__(
        self,
        endpoint: str,
        cookie: str,
        cluster: str,
        jobname: str,
        workspace: str,
        script: str,
        image: str,
        description: str = "",
        worker_num: int = 1,
        priority: int = 1,
        resource_dict: dict = None,
    ):
        self.endpoint = endpoint
        self.cookie = cookie
        self.cluster = cluster
        self.jobname = jobname
        if len(self.jobname) == 0 or len(self.jobname) > 40:
            raise ValueError("jobname must be non-empty and less than 40 characters")
        if not re.match(r"^[a-z0-9][a-z0-9-]+[a-z0-9]$", self.jobname):
            raise ValueError(
                "jobname only allows lowercase letters, numbers, and hyphens, "
                "and cannot start or end with a hyphen"
            )

        self.workspace = workspace
        self.script = script
        self.image = image
        self.description = description
        self.worker_num = worker_num
        assert self.worker_num > 0, "worker_num must be greater than 0"
        self.priority = priority
        assert self.priority in [0, 1, 2], "priority must be 0, 1, or 2"

        self.resource_dict = resource_dict
        if self.resource_dict is None:
            self.resource_dict = {
                "cpu": 1,
                "gpu": 1,
                "memory": "1Gi",
                "gpu_type": "nvidia.com/gpu",
                "share_memory": "1Gi",
                "ephemeral_storage": "1Gi",
            }

    def make_request_url(self) -> str:
        return f"{self.endpoint}/api/v1/workloads"

    def make_request_headers(self) -> dict:
        return {"Cookie": self.cookie, "Content-Type": "text/plain"}

    def convert_resource_dict(self) -> dict:
        return {
            "cpu": str(self.resource_dict["cpu"]),
            "gpu": str(self.resource_dict["gpu"]),
            "memory": self.resource_dict["memory"],
            "gpuType": self.resource_dict["gpu_type"],
            "shareMemory": self.resource_dict["share_memory"],
            "ephemeralStorage": self.resource_dict["ephemeral_storage"],
        }

    def make_request_data(self) -> dict:
        resources = []
        master_resource = {"name": "Master", "replica": 1, **self.convert_resource_dict()}
        resources.append(master_resource)
        if self.worker_num > 1:
            worker_resource = {
                "name": "Worker",
                "replica": self.worker_num - 1,
                **self.convert_resource_dict(),
            }
            resources.append(worker_resource)

        return json.dumps(
            {
                "workspace": self.workspace,
                "priority": self.priority,
                "displayName": self.jobname,
                "gvk": {"kind": "PyTorchJob"},
                "description": self.description,
                "entryPoint": self.script,
                "isSSHEnabled": True,
                "isSupervised": False,
                "isTolerateAll": False,
                "tolerations": {},
                "image": self.image,
                "maxRetry": 3,
                "resources": resources,
                "ttlSecondsAfterFinished": 0,
                "schedulerTime": 0,
                "customerLabels": {},
            }
        )


class PolarisGetWorkloadStatusRequest(PolarisRequest):
    method: str = "GET"

    def __init__(self, endpoint: str, cookie: str, workload_id: str):
        self.endpoint = endpoint
        self.cookie = cookie
        self.workload_id = workload_id

    def make_request_url(self) -> str:
        return f"{self.endpoint}/api/v1/workloads/{self.workload_id}"

    def make_request_headers(self) -> dict:
        return {
            "Cookie": self.cookie,
        }


class PolarisClient:
    def __init__(
        self,
        endpoint: str = None,
        cluster: str = None,
        user_xcs_id: str = None,
        user_xcs_password: str = None,
    ):
        """
        Polaris Client
        :param cluster: the cluster to initialize
        :param endpoint: the endpoint of the cluster
        :param user_xcs_id: the user id of the xcs platform
        :param user_xcs_password: the password of the xcs platform
        """
        assert endpoint is not None, "endpoint is required"
        assert cluster is not None, "cluster is required"
        self.endpoint = endpoint
        self.cluster = cluster
        assert user_xcs_id is not None, "user_xcs_id is required"
        assert user_xcs_password is not None, "user_xcs_password is required"
        self.user_xcs_id = user_xcs_id
        self.user_xcs_password = user_xcs_password

        self.logged_in_flag = False
        self.cookie = None

    @staticmethod
    def make_cookie_str(login_response_json: dict):
        return (
            f"Arsenal-UserName={urllib.parse.quote(login_response_json['userName'])}; "
            f"Arsenal-Token={urllib.parse.quote(login_response_json['token'], safe='')}; "
            f"Arsenal-UserId={login_response_json['userId']}; "
            f"Arsenal-Token-Expire={login_response_json['expire']}; "
            f"Arsenal-IsAdmin={'true' if login_response_json['isAdmin'] else 'false'}; "
            f"Arsenal-IsQueueAdmin={'true' if login_response_json['isQueueAdmin'] else 'false'}"
        )

    def is_logged_in(self):
        return self.logged_in_flag and self.cookie is not None

    def login(self):
        if self.is_logged_in():
            return self.cookie
        login_request = PolarisLoginRequest(
            user_xcs_id=self.user_xcs_id, user_xcs_password=self.user_xcs_password, endpoint=self.endpoint
        )
        login_response_json = login_request.send()
        cookie_str = self.make_cookie_str(login_response_json)
        self.logged_in_flag = True
        self.cookie = cookie_str
        return cookie_str

    def create_workload(
        self,
        workspace: str = "dev",
        priority: int = 1,
        jobname: str = "test",
        description: str = "test",
        script: str = "test",
        image: str = "test",
        resource_dict: dict = None,
        worker_num: int = 1,
    ):
        assert self.is_logged_in(), "Please login first"
        response = PolarisCreateWorkloadRequest(
            endpoint=self.endpoint,
            cookie=self.cookie,
            cluster=self.cluster,
            jobname=jobname,
            workspace=workspace,
            script=script,
            image=image,
            description=description,
            worker_num=worker_num,
            priority=priority,
            resource_dict=resource_dict,
        ).send()
        array_format_ids = response.get("workloadIds")
        if array_format_ids is None:
            return response.get("workloadId", None)
        else:
            if len(array_format_ids) == 0:
                return None
            else:
                return array_format_ids[0]

    def get_workload_status(self, workload_id: str):
        assert self.is_logged_in(), "Please login first"
        return (
            PolarisGetWorkloadStatusRequest(
                endpoint=self.endpoint, cookie=self.cookie, workload_id=workload_id
            )
            .send()
            .get("phase")
        )
