import gc
import json
import os
import sys
import uuid
from collections.abc import Callable
from typing import Any

import httpx
import torch
from comfy.utils import ProgressBar  # type: ignore
from signature_core.img.tensor_image import TensorImage
from signature_core.logger import console
from uuid_extensions import uuid7str

from .categories import UTILS_CAT
from .shared import BASE_COMFY_DIR, any_type

sys.path.append(BASE_COMFY_DIR)
import comfy  # type: ignore
import execution  # type: ignore


class PlaceholderServer:
    def __init__(self):
        self.client_id = str(uuid.uuid4())
        self.outputs = {}
        self.prompt = {}

    def add_on_prompt_handler(self, handler: Callable):
        pass

    def add_on_execution_start_handler(self, handler: Callable):
        pass

    def add_on_execution_end_handler(self, handler: Callable):
        pass

    def send_sync(self, event: str, data: dict[str, Any], sid: str | None = None):
        console.log(f"Send sync: {event}, {data}, {sid}")

    def send_sync_if_running(self, event: str, data: dict[str, Any], sid: str | None = None):
        console.log(f"Send sync if running: {event}, {data}, {sid}")

    def queue_prompt(self, prompt_id: str, prompt: dict[str, Any]):
        self.prompt = prompt

    def get_images(self) -> list[str]:
        return []

    def get_history(self) -> list[dict[str, Any]]:
        return []

    def get_prompt(self) -> dict[str, Any]:
        return self.prompt


server = PlaceholderServer()
executor = execution.PromptExecutor(server)  # type: ignore


class Workflow:
    def __init__(self, workflow: dict):
        self.__workflow = workflow

    @property
    def data(self) -> dict:
        return self.__workflow

    def get_inputs(self) -> dict:
        data = {}
        for key, value in self.__workflow.items():
            if value.get("class_type", "").startswith("signature_input"):
                data_value = value.get("inputs") or None
                if data_value is None:
                    continue
                data_title = data_value.get("title")
                data_type = (data_value.get("subtype") or "").upper()
                data_value = data_value.get("value")
                if data_value == "":
                    data_value = None
                data.update({key: {"title": data_title, "type": data_type, "value": data_value}})
        return data

    def get_outputs(self) -> dict:
        data = {}
        for key, value in self.__workflow.items():
            if value.get("class_type", "").startswith("signature_output"):
                data_value = value.get("inputs") or None
                if data_value is None:
                    continue
                data_title = data_value.get("title")
                data_type = (data_value.get("subtype") or "").upper()
                data_metadata = data_value.get("metadata")
                data.update({key: {"title": data_title, "type": data_type, "metadata": data_metadata}})
        return data

    def set_inputs(self, new_inputs: dict):
        for key, value in self.__workflow.items():
            if value.get("class_type", "").startswith("signature_input"):
                old_inputs = value.get("inputs") or None
                if old_inputs is None:
                    continue
                for new_input_key in new_inputs.keys():
                    if new_input_key != key:
                        continue
                    new_input_dict = new_inputs[new_input_key]
                    for new_input_key, new_input_value in new_input_dict.items():
                        value["inputs"][new_input_key] = new_input_value

    def set_input(self, key: str, value: Any):
        self.set_inputs({key: value})


class Wrapper:

    def upload_file(self, base_url, file_path: str, token: str) -> dict:
        url = f"{base_url}/assets/upload"
        headers = {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Authorization": "Bearer " + token,
        }
        output = {}
        with httpx.Client() as client:
            try:
                with open(file_path, "rb") as file:
                    files = {"file": file}
                    response = client.post(url, headers=headers, files=files, timeout=3)
                    if response.status_code == 200:
                        buffer = response.content
                        output = json.loads(buffer)
            except Exception as e:
                console.log(e)
        return output

    def get_workflow(self, base_url: str, workflow_id: str, token: str) -> dict:
        url = f"{base_url}/workflows/{workflow_id}"
        headers = {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Authorization": "Bearer " + token,
        }
        output = {}
        with httpx.Client() as client:
            try:
                response = client.get(url, headers=headers, timeout=3)
                if response.status_code == 200:
                    buffer = response.content
                    output = json.loads(buffer)
            except Exception as e:
                console.log(e)
        return output

    def run_workflow_job(self, base_url, org_id: str, workflow: Workflow, token: str) -> list:
        def process_data_chunk(chunk: str, nodes_ids: list) -> dict | None:
            cleaned_chunk = chunk.strip()
            if cleaned_chunk.startswith("data:"):
                cleaned_chunk = cleaned_chunk[len("data:") :].strip()

            try:
                data = json.loads(cleaned_chunk)
                data_type = data.get("type")
                if data_type == "execution_cached":
                    cached_nodes_ids = data["data"].get("nodes")
                    remaining_ids = list(set(nodes_ids) - set(cached_nodes_ids))
                    return {"remaining_ids": remaining_ids}
                if data_type == "executing":
                    node_id = data["data"].get("node")
                    remaining_ids = list(set(nodes_ids) - {node_id})
                    return {"remaining_ids": remaining_ids}
                if data_type == "executed":
                    output = data["data"].get("output")
                    if output is None:
                        return None
                    if "signature_output" in output:
                        signature_output = output["signature_output"]
                        # console.log(f"Signature output: {signature_output}")
                        for item in signature_output:
                            title = item.get("title")
                            value = item.get("value")
                            # console.log(f"======== Title: {title}, Value: {value}")
                            return {"output": {"title": title, "value": value}}
                elif data_type == "execution_error":
                    error = data["data"]
                    return {"error": error}
                elif data_type == "status":
                    status = data["data"].get("status")
                    if isinstance(status, dict):
                        queue_remaining = status.get("exec_info", None).get("queue_remaining")
                        return {"queue": max(0, int(queue_remaining) - 1)}
                else:
                    return {"waiting": "no communication yet"}
                    # elif status == "finished":
                    #     # console.log("Workflow finished!")
                    #     return {"finished": True}

            except json.JSONDecodeError:
                console.log(f"JSON decode error: {cleaned_chunk}")
            return None

        url = f"{base_url}/generate-signature"
        # queue_url = f"{base_url}/queue"

        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "X-Org-Id": org_id,
        }
        total_nodes = list(workflow.data.keys())
        total_steps = len(total_nodes)
        remaining_ids = total_nodes
        workflow_data = json.dumps(workflow.data)
        params = {"data": {"client_id": uuid7str(), "workflow": workflow_data}}
        outputs = []
        try:
            data = json.dumps(params)
            pbar = ProgressBar(total_steps)  # type: ignore
            with httpx.Client() as client:
                prompt_id = client.post(url, data=data, headers=headers).json()  # type: ignore
                # queue = client.get(queue_url, headers=headers).json()
                # console.log(f"Queue: {queue}")
                stream_url = f"{url}/{prompt_id}"
                base_retrys = 20
                retrys = base_retrys
                with client.stream(method="GET", url=stream_url, headers=headers, timeout=9000000) as stream:
                    for chunk in stream.iter_lines():
                        if not chunk:
                            continue

                        result = process_data_chunk(chunk, remaining_ids)
                        if result is None:
                            retrys -= 1
                            console.log(f"Remaining retrys: {retrys}")
                        elif "error" in result:
                            error = result["error"]
                            raise Exception(error)
                        elif "output" in result:
                            retrys = base_retrys
                            outputs.append(result["output"])
                        elif "remaining_ids" in result:
                            retrys = base_retrys
                            remaining_ids = result["remaining_ids"]
                            step = total_steps - len(remaining_ids)
                            pbar.update_absolute(step, total_steps)
                        elif "queue" in result:
                            queue = result["queue"]
                            console.log(f"Queue: {queue}")
                        elif "waiting" in result:
                            retrys -= 1
                            console.log(f"Remaining retrys: {retrys}")
                        if retrys <= 0:
                            raise Exception("Failed to get communication")
                pbar.update_absolute(total_steps, total_steps)

        except httpx.HTTPStatusError as exc:
            console.log(f"HTTP error occurred: {exc}")
            # raise Exception(exc)
        except Exception as exc:
            console.log(f"HTTP error occurred: {exc}")
            # raise Exception(exc)
        return outputs

    def upload_image_to_s3(self, base_url, token: str, image: TensorImage) -> str | None:
        url = f"{base_url}/assets/upload"  # Replace with actual URL
        headers = {"Authorization": f"Bearer {token}", "Access-Control-Allow-Origin": "*"}

        query_params = {"id": uuid7str(), "prefix": ""}
        image_bytes = image.get_bytes("png")

        files = {
            "file": (
                f"{uuid7str()}.png",
                image_bytes,
                "image/png",
            )
        }

        try:
            response = httpx.post(url, params=query_params, files=files, headers=headers)

            if response.status_code == 200:
                return response.json()

            print(f"Failed with status code {response.status_code}")
            print("Response:", response.json())

        except httpx.RequestError as exc:
            print(f"An error occurred while requesting: {exc}")

        return None

    def process_outputs(self, job_outputs, node_outputs):

        def process_data(node_output: dict, job_output: dict):

            node_type = node_output.get("type")
            value = job_output.get("value")
            if value is None or not isinstance(node_type, str):
                return []
            if node_type in ("IMAGE", "MASK"):
                if not isinstance(value, str):
                    return None
                image_path = os.path.join(BASE_COMFY_DIR, "output", value)
                output_image = TensorImage.from_local(image_path)
                return output_image.get_BWHC()
            if node_type == "INT":
                return int(value)
            if node_type == "FLOAT":
                return float(value)
            if node_type == "BOOLEAN":
                return bool(value)
            return str(value)

        outputs = []
        for node_output in node_outputs:
            for job_output in job_outputs:
                if not isinstance(job_output, dict) or not isinstance(node_output, dict):
                    continue
                node_name = node_output.get("title") or []
                job_name = job_output.get("title") or []
                if isinstance(node_name, str):
                    node_name = [node_name]
                if isinstance(job_name, str):
                    job_name = [job_name]
                # console.log(f"Node name: {node_name}, Job name: {job_name}")
                for node_name_part in node_name:
                    for job_name_part in job_name:
                        if node_name_part != job_name_part:
                            continue
                        data = process_data(node_output, job_output)
                        if data is None:
                            continue
                        outputs.append(data)
                        break
                # console.log(f"Added {node_name} {node_type}")
        # console.log(f"=====================>>> Node outputs: {len(outputs)}")
        return tuple(outputs)

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        inputs = {
            "required": {
                "data": ("STRING", {"default": ""}),
            },
            "optional": {},
        }

        return inputs

    RETURN_TYPES = (any_type,) * 20
    FUNCTION = "execute"
    CATEGORY = UTILS_CAT

    def execute(self, **kwargs):

        data = kwargs.get("data")
        # console.log(f"kwargs: {kwargs}")

        fallback = (None,) * 20
        if data is None:
            return fallback
        json_data = json.loads(data)
        if not isinstance(json_data, dict):
            return fallback
        base_url = json_data.get("origin") or None
        workflow_api = json_data.get("workflow_api") or None
        token = json_data.get("token") or None
        inference_host = json_data.get("inference_host") or None
        widget_inputs = json_data.get("widget_inputs") or []
        # console.log(f"Widget inputs: {widget_inputs}")

        if inference_host is None or inference_host == "":
            inference_host = base_url
        # console.log(f"Origin: {base_url}, Inference host: {inference_host}")
        if not isinstance(base_url, str):
            return fallback
        if not isinstance(workflow_api, dict):
            return fallback
        if not isinstance(token, str):
            return fallback

        workflow = Workflow(workflow_api)
        node_inputs = workflow.get_inputs()
        # console.log(f"Node inputs: {node_inputs}")
        workflow_outputs = workflow.get_outputs()
        output_ids = workflow_outputs.keys()
        node_outputs = workflow_outputs.values()
        # console.log(f"Node outputs: {node_outputs}")

        for key, value in node_inputs.items():
            if not isinstance(value, dict):
                continue
            node_title = value.get("title")
            node_type = value.get("type")
            if not isinstance(node_type, str) or not isinstance(node_title, str):
                continue
            comfy_value = kwargs.get(node_title)
            if comfy_value is None:
                for widget_input in widget_inputs:
                    if widget_input.get("title") == node_title:
                        widget_value = widget_input.get("value")
                        if widget_value is None:
                            continue
                        comfy_value = widget_input.get("value")
                        break
            if comfy_value is None:
                continue
            if node_type in ("IMAGE", "MASK"):
                if isinstance(comfy_value, torch.Tensor):
                    tensor_image = TensorImage.from_BWHC(comfy_value).get_base64()
                    # uploaded_image = self.upload_image_to_s3(base_url, token, tensor_image)
                    value.update({"value": tensor_image})
                    node_inputs[key] = value
            else:
                value.update({"value": comfy_value})
                node_inputs[key] = value

        workflow.set_inputs(node_inputs)
        workflow_api = workflow.data

        if not isinstance(workflow_api, dict):
            return fallback

        executor.execute(workflow_api, uuid.uuid4(), {}, output_ids)
        executor.reset()
        gc.collect()
        comfy.model_management.unload_all_models()
        comfy.model_management.cleanup_models()
        comfy.model_management.soft_empty_cache()

        if executor.success:
            console.log("Success wrapper inference")
        else:
            console.log("Failed wrapper inference")
            final_status = executor.status_messages[-1]
            console.log(f"Final status: {final_status}")
            if isinstance(final_status, dict):
                final_error = final_status.get("execution_error") or None
                if final_error is not None:
                    raise Exception(final_error)
        outputs = executor.history_result["outputs"].values()
        job_outputs = []
        for job_output in outputs:
            for key, value in job_output.items():
                if key == "signature_output":
                    job_outputs.extend(value)

        return self.process_outputs(job_outputs, node_outputs)
