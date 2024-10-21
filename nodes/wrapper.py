import json

import httpx
import torch
from comfy.utils import ProgressBar  # type: ignore
from signature_core.img.tensor_image import TensorImage
from signature_core.logger import console
from uuid_extensions import uuid7str

from .categories import UTILS_CAT
from .shared import any_type


class PlatfromWrapper:

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

    def run_workflow_job(self, base_url, org_id: str, json_data: dict, token: str) -> list:
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

        total_nodes = self.get_workflow_ids(json_data)
        total_steps = len(total_nodes)
        remaining_ids = total_nodes
        workflow = json.dumps(json_data)
        params = {"data": {"client_id": uuid7str(), "workflow": workflow}}
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

    def get_workflow_ids(self, json_data: dict) -> list:
        ids = []
        if not isinstance(json_data, dict):
            return None

        for value, _ in json_data.items():
            ids.append(value)

        return ids

    def update_workflow_inputs(self, json_data: dict, inputs: list) -> dict | None:

        if not isinstance(json_data, dict):
            return None

        for _, value in json_data.items():
            if value.get("class_type", "").startswith("signature_input"):
                value_inputs = value.get("inputs", {})
                for data in inputs:
                    input_title = data.get("title")
                    input_type = data.get("type")
                    value_title = value_inputs.get("title")
                    value_type = (value_inputs.get("subtype") or "").upper()
                    if input_title == value_title and input_type == value_type:
                        # console.log(f"Updating input: {value}")
                        # console.log(f"data : {data}")
                        value["inputs"]["value"] = data.get("value") or ""
        return json_data

    def get_workflow_outputs(self, json_data: dict) -> list:
        outputs = []

        if not isinstance(json_data, dict):
            return outputs

        for _, value in json_data.items():
            if value.get("class_type", "").startswith("signature_output"):
                value_inputs = value.get("inputs") or None
                if value_inputs is None:
                    continue
                output_title = value_inputs.get("title")
                output_type = (value_inputs.get("subtype") or "").upper()
                outputs.append({"title": output_title, "type": output_type})
        return outputs

    def get_workflow_inputs(self, json_data: dict) -> list:
        inputs = []

        if not isinstance(json_data, dict):
            return inputs

        for _, value in json_data.items():
            if value.get("class_type", "").startswith("signature_input"):
                value_inputs = value.get("inputs") or None
                if value_inputs is None:
                    continue
                input_title = value_inputs.get("title")
                input_type = (value_inputs.get("subtype") or "").upper()
                inputs.append({"title": input_title, "type": input_type})
        return inputs

    def process_outputs(self, job_outputs, node_outputs):

        def process_data(node_output: dict, job_output: dict):

            node_type = node_output.get("type")
            value = job_output.get("value")
            if value is None or not isinstance(node_type, str):
                return []
            if node_type in ("IMAGE", "MASK"):
                if not isinstance(job_output, dict):
                    return None
                console.log(f"value: {value}")
                if not isinstance(value, str):
                    return None
                if not value.startswith("https://") and not value.startswith("http://"):
                    try:
                        url = "https://mango-taskforce.signature.ai/view?filename=" + value
                        output_image = TensorImage.from_web(url)
                    except Exception:
                        url = "https://plugins-platform.signature.ai/view?filename=" + value
                        output_image = TensorImage.from_web(url)
                else:
                    output_image = TensorImage.from_web(value)
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
        workflow_id = json_data.get("workflow_id") or None
        token = json_data.get("token") or None
        inference_host = json_data.get("inference_host") or None
        widget_inputs = json_data.get("widget_inputs") or []
        # console.log(f"Widget inputs: {widget_inputs}")

        if inference_host is None or inference_host == "":
            inference_host = base_url
        # console.log(f"Origin: {base_url}, Inference host: {inference_host}")
        if not isinstance(base_url, str):
            return fallback
        if not isinstance(workflow_id, str):
            return fallback
        if not isinstance(token, str):
            return fallback

        result = self.get_workflow(base_url, workflow_id, token)

        if result is None:
            return fallback
        wf_api_string = result.get("workflow_api") or ""
        projects = result.get("projects") or []
        if len(projects) == 0:
            return fallback
        # org_id = projects[0].get("organisation") or None
        org_id = "66f16ff117e9f62c90b83e7f"
        if org_id is None:
            return fallback

        json_data = json.loads(wf_api_string)
        node_inputs = self.get_workflow_inputs(json_data)
        # console.log(f"Node inputs: {node_inputs}")
        node_outputs = self.get_workflow_outputs(json_data)
        # console.log(f"Node outputs: {node_outputs}")

        for node_input in node_inputs:
            node_title = node_input.get("title")
            node_type = node_input.get("type")
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
                    tensor_image = TensorImage.from_BWHC(comfy_value)
                    uploaded_image = self.upload_image_to_s3(base_url, token, tensor_image)
                    node_input.update({"value": uploaded_image})
            else:
                # console.log(f"Title: {node_title} Value: {comfy_value} Type: {node_type}")
                node_input.update({"value": comfy_value})

        json_data = self.update_workflow_inputs(json_data, node_inputs)

        if not isinstance(json_data, dict):
            return fallback

        job_outputs = self.run_workflow_job(inference_host, org_id, json_data, token)

        return self.process_outputs(job_outputs, node_outputs)


NODE_CLASS_MAPPINGS = {
    "signature_wrapper": PlatfromWrapper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "signature_wrapper": "SIG Wrapper",
}
