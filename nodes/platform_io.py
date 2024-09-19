import json
import os
import time
from datetime import datetime

import httpx
import torch
from signature_core.connectors.google_connector import GoogleConnector
from signature_core.functional.transform import cutout
from signature_core.img.tensor_image import TensorImage
from signature_core.logger import console
from uuid_extensions import uuid7str

from .categories import PLATFROM_IO_CAT
from .shared import BASE_COMFY_DIR, any_type


class PlatformInputImage:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "title": ("STRING", {"default": "Input Image"}),
                "subtype": (["image", "mask"],),
                "required": ("BOOLEAN", {"default": True}),
                "include_alpha": ("BOOLEAN", {"default": False}),
                "value": ("STRING", {"default": ""}),
                "metadata": ("STRING", {"default": "{}", "multiline": True}),
            },
            "optional": {
                "fallback": (any_type,),
            },
        }

    RETURN_TYPES = (any_type,)
    FUNCTION = "apply"
    CATEGORY = PLATFROM_IO_CAT
    OUTPUT_IS_LIST = (True,)

    def apply(
        self,
        value,
        title: str,
        metadata: str,
        subtype: str,
        required: bool,
        include_alpha: bool,
        fallback=None,
    ):
        def post_process(output: TensorImage, include_alpha: bool) -> TensorImage:
            if include_alpha is False and output.shape[1] == 4:
                # get alpha
                rgb = TensorImage(output[:, :3, :, :])
                alpha = TensorImage(output[:, -1, :, :])
                output, _ = cutout(rgb, alpha)
            return output

        if "," in value:
            value = value.split(",")
        else:
            value = [value] if value != "" else []
        outputs: list[TensorImage | torch.Tensor] = []
        for i, _ in enumerate(value):
            item = value[i]
            if isinstance(item, str):
                if item != "":
                    if item.startswith("http"):
                        output = TensorImage.from_web(item)
                    else:
                        try:
                            output = TensorImage.from_base64(item)
                        except:
                            raise ValueError(f"Unsupported input format: {item}")
                    outputs.append(output)
        if len(outputs) == 0:
            if fallback is None:
                raise ValueError("No input found")
            tensor_fallback = TensorImage.from_BWHC(fallback)
            outputs.append(tensor_fallback)
        for i, _ in enumerate(outputs):
            output = outputs[i]
            if isinstance(output, torch.Tensor):
                output = TensorImage(output)
            if subtype == "mask":
                outputs[i] = output.get_grayscale().get_BWHC()
            else:
                if isinstance(output, TensorImage):
                    outputs[i] = post_process(output, include_alpha).get_BWHC()
        return (outputs,)


class PlatformInputConnector:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "title": ("STRING", {"default": "Input Connector"}),
                "subtype": (["google_drive"],),
                "required": ("BOOLEAN", {"default": True}),
                "override": ("BOOLEAN", {"default": False}),
                "token": ("STRING", {"default": ""}),
                "mime_type": ("STRING", {"default": "image/png"}),
                "value": ("STRING", {"default": ""}),
                "metadata": ("STRING", {"default": "{}", "multiline": True}),
            },
        }

    RETURN_TYPES = ("FILE",)
    FUNCTION = "apply"
    CATEGORY = PLATFROM_IO_CAT

    def apply(
        self,
        value: str,
        token: str,
        mime_type: str,
        override: bool,
        title: str,
        metadata: str,
        subtype: str,
        required: bool,
    ):
        connector = GoogleConnector(token=token)
        input_folder = os.path.join(BASE_COMFY_DIR, "input")
        data = connector.download(id=value, mime_type=mime_type, root_path=input_folder, override=override)
        return (data,)


class PlatformInputText:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "title": ("STRING", {"default": "Input Text"}),
                "subtype": (["string", "positive_prompt", "negative_prompt"],),
                "required": ("BOOLEAN", {"default": True}),
                "value": ("STRING", {"multiline": True, "default": ""}),
                "metadata": ("STRING", {"default": "{}", "multiline": True}),
            },
            "optional": {
                "fallback": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "apply"
    CATEGORY = PLATFROM_IO_CAT

    def apply(self, value: str, title: str, metadata: str, subtype: str, required: bool, fallback=None):
        if value == "":
            value = fallback or ""
        return (value,)


class PlatformInputNumber:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "title": ("STRING", {"default": "Input Number"}),
                "subtype": (["float", "int"],),
                "required": ("BOOLEAN", {"default": True}),
                "value": ("FLOAT", {"default": 0}),
                "metadata": ("STRING", {"default": "{}", "multiline": True}),
            },
        }

    RETURN_TYPES = (any_type,)
    FUNCTION = "apply"
    CATEGORY = PLATFROM_IO_CAT

    def apply(self, value: float, title: str, metadata: str, subtype: str, required: bool):
        if subtype == "int":
            value = int(value)
        else:
            value = float(value)
        return (value,)


class PlatformInputBoolean:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "title": ("STRING", {"default": "Input Boolean"}),
                "subtype": (["boolean"],),
                "required": ("BOOLEAN", {"default": True}),
                "value": ("BOOLEAN", {"default": False}),
                "metadata": ("STRING", {"default": "{}", "multiline": True}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "apply"
    CATEGORY = PLATFROM_IO_CAT

    def apply(self, value: bool, title: str, subtype: str, metadata: str, required: bool):
        return (value,)


class PlatformInputSlider:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "title": ("STRING", {"default": "Input Slider"}),
                "subtype": (["float", "int"],),
                "required": ("BOOLEAN", {"default": True}),
                "value": ("FLOAT", {"default": 0}),
                "min_value": ("FLOAT", {"default": 0}),
                "max_value": ("FLOAT", {"default": 10}),
                "metadata": ("STRING", {"default": "{}", "multiline": True}),
            },
        }

    RETURN_TYPES = (any_type,)
    FUNCTION = "apply"
    CATEGORY = PLATFROM_IO_CAT

    def apply(
        self,
        value: float,
        min_value: float,
        max_value: float,
        title: str,
        metadata: str,
        subtype: str,
        required: str,
    ):
        if subtype == "int":
            value = max(min(int(max_value), int(value)), int(min_value))
        else:
            value = max(min(max_value, value), min_value)
        return (value,)


class PlatformOutput:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "title": ("STRING", {"default": "Output Image"}),
                "subtype": (["image", "mask", "int", "float", "string", "dict"],),
                "metadata": ("STRING", {"default": "", "multiline": True}),
                "value": (any_type,),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    INPUT_IS_LIST = True
    FUNCTION = "apply"
    CATEGORY = PLATFROM_IO_CAT

    def __save_outputs(
        self, img, title: str, subtype: str, thumbnail_size: int, output_dir: str, metadata: str = ""
    ) -> dict | None:
        random_str = str(torch.randint(0, 100000, (1,)).item())
        current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"signature_{current_time_str}_{random_str}.png"
        save_path = os.path.join(output_dir, file_name)

        output_img = TensorImage(img)

        thumbnail_img = output_img.get_resized(thumbnail_size)
        thumbnail_path = save_path.replace(".png", "_thumbnail.jpeg")
        thumbnail_file_name = file_name.replace(".png", "_thumbnail.jpeg")
        thumbnail_saved = thumbnail_img.save(thumbnail_path)

        image_saved = output_img.save(save_path)

        if image_saved and thumbnail_saved:
            return {
                "title": title,
                "type": subtype,
                "metadata": metadata,
                "value": file_name,
                "thumbnail": thumbnail_file_name if thumbnail_saved else None,
            }

        return None

    def apply(self, value, title: str, subtype: str, metadata: str = ""):
        if len(subtype) == 0 or len(value) == 0:
            raise ValueError("No input found")
        main_subtype = subtype[0]
        supported_types = ["image", "mask", "int", "float", "string", "dict"]
        if main_subtype not in supported_types:
            raise ValueError(f"Unsupported output type: {subtype}")

        output_dir = os.path.join(BASE_COMFY_DIR, "output")
        results = []
        thumbnail_size = 1024
        # console.log(f"Input size {len(value)}")
        for item in value:
            if isinstance(item, torch.Tensor):
                if main_subtype in ["image", "mask"]:
                    tensor_images = TensorImage.from_BWHC(item.to("cpu"))
                    for img in tensor_images:
                        # console.log(f"Input tensor shape {img.shape}")
                        result = self.__save_outputs(
                            img, title, main_subtype, thumbnail_size, output_dir, metadata
                        )
                        if result:
                            results.append(result)
                else:
                    raise ValueError(f"Unsupported output type: {type(item)}")
            else:
                value_json = json.dumps(item) if main_subtype == "dict" else item
                results.append(
                    {"title": title, "type": main_subtype, "metadata": metadata, "value": value_json}
                )

        # console.log(f"Output size {len(results)}")
        # console.log(f"Output {results}")
        return {"ui": {"signature_output": results}}


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

    def run_workflow_job(self, base_url, org_id: str, workflow: str, token: str) -> list:
        def process_data_chunk(chunk: str) -> dict | None:
            cleaned_chunk = chunk.strip()
            if cleaned_chunk.startswith("data:"):
                cleaned_chunk = cleaned_chunk[len("data:") :].strip()

            try:
                data = json.loads(cleaned_chunk)
                data_type = data.get("type")
                if data_type == "executed":
                    output = data["data"].get("output")
                    if output is None:
                        return None
                    if "signature_output" in output:
                        signature_output = output["signature_output"]
                        console.log(f"Signature output: {signature_output}")
                        for item in signature_output:
                            title = item.get("title")
                            value = item.get("value")
                            console.log(f"======== Title: {title}, Value: {value}")
                            return {"title": title, "value": value}

                elif data_type == "status" and data["data"].get("status") == "finished":
                    console.log("Workflow finished!")

            except json.JSONDecodeError:
                console.log(f"JSON decode error: {cleaned_chunk}")
            return None

        url = f"{base_url}/generate-signature"

        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "X-Org-Id": org_id,
        }

        params = {"data": {"client_id": uuid7str(), "workflow": workflow}}
        data = json.dumps(params)
        outputs = []
        retry_attempts = 5
        delay_between_attempts = 5  # seconds

        try:
            with httpx.Client() as client:
                response = client.post(url, data=data, headers=headers)  # type: ignore
                prompt_id = response.json()
                stream_url = f"{url}/{prompt_id}"
                for _ in range(retry_attempts):
                    with client.stream(
                        method="GET", url=stream_url, headers=headers, timeout=900000
                    ) as stream:
                        for chunk in stream.iter_lines():
                            if not chunk:
                                continue
                            console.log(f"CHUNK: {chunk}")
                            result = process_data_chunk(chunk)
                            if result is not None:
                                outputs.append(result)
                    break  # Exit the loop if successful
        except httpx.HTTPStatusError as exc:
            console.log(f"HTTP error occurred: {exc}")
        except httpx.RequestError as _:
            time.sleep(delay_between_attempts)
        return outputs

    def upload_image_to_s3(self, base_url, token: str, image: TensorImage) -> str | None:
        url = f"{base_url}/assets/upload"  # Replace with actual URL
        headers = {"Authorization": f"Bearer {token}", "Access-Control-Allow-Origin": "*"}

        query_params = {"id": uuid7str(), "prefix": ""}
        image_bytes = image.get_bytes("png")

        # Open the image file in binary mode
        files = {
            "file": (
                f"{uuid7str()}.png",
                image_bytes,
                "image/png",
            )  # Adjust content type if needed (e.g., for JPEG: "image/jpeg")
        }

        # Make the POST request to upload the image
        try:
            response = httpx.post(url, params=query_params, files=files, headers=headers)

            # Handle response
            if response.status_code == 200:
                return response.json()

            print(f"Failed with status code {response.status_code}")
            print("Response:", response.json())

        except httpx.RequestError as exc:
            print(f"An error occurred while requesting: {exc}")

        return None

    def update_workflow_inputs(self, json_string: str, inputs: list) -> str | None:
        json_data = json.loads(json_string)
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
                        value["inputs"]["value"] = data.get("value") or ""
        return json.dumps(json_data)

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "data": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 10000000000000000}),
            },
        }

    RETURN_TYPES = (any_type,) * 20
    FUNCTION = "process"
    CATEGORY = PLATFROM_IO_CAT

    def process(self, **kwargs):
        data = kwargs.get("data")
        fallback = (None,) * 20
        if data is None:
            return fallback
        json_data = json.loads(data)
        if not isinstance(json_data, dict):
            return fallback
        base_url = json_data.get("origin") or None
        workflow_id = json_data.get("workflow_id") or None
        token = json_data.get("token") or None

        if not isinstance(base_url, str):
            return fallback
        if not isinstance(workflow_id, str):
            return fallback
        if not isinstance(token, str):
            return fallback

        result = self.get_workflow(base_url, workflow_id, token)
        # console.log(result)
        if result is None:
            return fallback
        workflow_api_string = result.get("workflow_api") or ""
        projects = result.get("projects") or []
        if len(projects) == 0:
            return fallback
        org_id = projects[0].get("organisation") or None
        if org_id is None:
            return fallback

        node_inputs = json_data.get("inputs") or []
        for node_input in node_inputs:
            node_title = node_input.get("title")
            node_type = node_input.get("type")
            if node_type in ("IMAGE", "MASK"):
                comfy_image = kwargs.get(node_title)
                if isinstance(comfy_image, torch.Tensor):
                    tensor_image = TensorImage.from_BWHC(comfy_image)
                    uploaded_image = self.upload_image_to_s3(base_url, token, tensor_image)
                    node_input.update({"value": uploaded_image})
            else:
                node_input.update({"value": kwargs.get(node_title)})

        node_outputs = json_data.get("outputs") or []
        updated_workflow_api_string = self.update_workflow_inputs(workflow_api_string, node_inputs)
        if not isinstance(updated_workflow_api_string, str):
            return fallback

        job_outputs = self.run_workflow_job(base_url, org_id, updated_workflow_api_string, token)

        outputs = []
        console.log(f"=====================>>> Node outputs: {node_outputs}")
        console.log(f"=====================>>> Job outputs: {job_outputs}")

        for node_output in node_outputs:
            for job_output in job_outputs:
                node_name = node_output.get("title")
                node_type = node_output.get("type")
                job_name = job_output.get("title")
                if node_name != job_name:
                    continue
                if node_type in ("IMAGE", "MASK"):
                    base_value = job_output.get("value")
                    # need to ask anderson for this
                    if not base_value.startswith("https://"):
                        try:
                            base_value = "https://admin.signature.ai/view?filename=" + base_value
                        except Exception:
                            base_value = "https://plugins-platform.signature.ai/view?filename=" + base_value
                    console.log(f"=====================>>> Base value modified: {base_value}")
                    outputs.append(TensorImage.from_web(base_value).get_BWHC())
                elif node_type == "INT":
                    outputs.append(int(job_output.get("value")))
                elif node_type == "FLOAT":
                    outputs.append(float(job_output.get("value")))
                else:
                    outputs.append(job_output.get("value"))
                # console.log(f"Added {node_name} {node_type}")
        return tuple(outputs)


NODE_CLASS_MAPPINGS = {
    "signature_input_image": PlatformInputImage,
    "signature_input_text": PlatformInputText,
    "signature_input_number": PlatformInputNumber,
    "signature_input_boolean": PlatformInputBoolean,
    "signature_input_slider": PlatformInputSlider,
    "signature_input_connector": PlatformInputConnector,
    "signature_wrapper": PlatfromWrapper,
    "signature_output": PlatformOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "signature_input_image": "SIG Input Image",
    "signature_input_text": "SIG Input Text",
    "signature_input_number": "SIG Input Number",
    "signature_input_boolean": "SIG Input Boolean",
    "signature_input_slider": "SIG Input Slider",
    "signature_input_connector": "SIG Input Connector",
    "signature_wrapper": "SIG Wrapper",
    "signature_output": "SIG Output",
}
