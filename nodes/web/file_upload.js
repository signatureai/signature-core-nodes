import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";


const NODES = {
    "Load File": "file",
    "Load Folder": "folder",
};

async function uploadFile(file) {
    //TODO: Add uploaded file to cache with Cache.put()?
    try {
        // Wrap file in formdata so it includes filename
        const body = new FormData();
        const i = file.webkitRelativePath.lastIndexOf('/');
        const subfolder = file.webkitRelativePath.slice(0,i+1)
        const new_file = new File([file], file.name, {
            type: file.type,
            lastModified: file.lastModified,
        });
        body.append("image", new_file);
        if (i > 0) {
            body.append("subfolder", subfolder);
        }
        const resp = await api.fetchApi("/upload/image", {
            method: "POST",
            body,
        });

        if (resp.status === 200) {
            return resp.status
        } else {
            alert(resp.status + " - " + resp.statusText);
        }
    } catch (error) {
        alert(error);
    }
}

function chainCallback(object, property, callback) {
    if (object == undefined) {
        //This should not happen.
        console.error("Tried to add callback to non-existant object")
        return;
    }
    if (property in object) {
        const callback_orig = object[property]
        object[property] = function () {
            const r = callback_orig.apply(this, arguments);
            callback.apply(this, arguments);
            return r
        };
    } else {
        object[property] = callback;
    }
}


async function loadFile(file, pathValueWidget) {
    if (await uploadFile(file) != 200) {
        // Upload failed and file cannot be added to options
        return;
    }
    const result = {'name': file.name, 'type': file.type};
    const resultString = JSON.stringify(result);
    const value = pathValueWidget.value;
    if (value == '') {
        pathValueWidget.value = resultString;
    } else {
        pathValueWidget.value = value + "&&" + resultString;
    }
}

function addUploadWidget(nodeType, valueWidget, type="file", accept="*/*") {
    chainCallback(nodeType.prototype, "onNodeCreated", function() {
        const pathValueWidget = this.widgets.find((w) => w.name === valueWidget);
        const fileInput = document.createElement("input");
        chainCallback(this, "onRemoved", () => {
            fileInput?.remove();
        });

        if (type == "folder") {
            Object.assign(fileInput, {
                type: "file",
                style: "display: none",
                webkitdirectory: true,
                onchange: async () => {
                    if (fileInput.files.length) {
                        for (const file of fileInput.files) {
                            await loadFile(file, pathValueWidget);
                        }
                    }
                },
            });
        } else if (type == "file") {
            Object.assign(fileInput, {
                type: "file",
                accept: accept,
                style: "display: none",
                onchange: async () => {
                    if (fileInput.files.length) {
                        for (const file of fileInput.files) {
                            await loadFile(file, pathValueWidget);
                        }
                    }
                },
            });
        } else {
            throw "Unknown upload type"
        }

        document.body.append(fileInput);
        let uploadWidget = this.addWidget("button", "choose " + type + " to upload", "image", () => {
            //clear the active click event
            app.canvas.node_widget = null
            fileInput.click();
        });
        uploadWidget.options.serialize = false;
    });
}


const ext = {
    name: "signature.file_upload",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const title = nodeData?.name;
        if (NODES.hasOwnProperty(title)) {
            addUploadWidget(nodeType, "value", NODES[title], "*/*");
        }
    }
};

app.registerExtension(ext);