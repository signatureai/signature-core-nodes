import { app } from "../../../scripts/app.js";

const origin = window.location.search;
const urlParams = new URLSearchParams(origin);
const env = urlParams.get("env");
let token = urlParams.get("token");
let main_url = "https://api-staging.signature.ai/api/v1";
if (env === "prod") {
  main_url = "https://api.signature.ai/api/v1";
}

if (token == undefined || token == "") {
  function getCookie(name) {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop().split(";").shift();
  }
  token = getCookie("accessToken");
}

const headers = getHeaders(token);

const findWidgetByName = (node, name) => {
  return node.widgets ? node.widgets.find((w) => w.name === name) : null;
};

function getHeaders(token) {
  const headers = new Headers();
  headers.append("Content-Type", "application/json");
  headers.append("Access-Control-Allow-Origin", "*");
  headers.append("Authorization", `Bearer ${token}`);
  return headers;
}

async function getOrganisationsData() {
  let url = main_url + "/organisations?skip=0&limit=1000";
  const response = await fetch(url, {
    method: "GET",
    headers: headers,
  });
  let data = await response.json();
  return data;
}

async function getProjectsData(organisationId) {
  let url =
    main_url + "/projects?organisationId=" + organisationId + "&skip=0&limit=1000";
  const response = await fetch(url, {
    method: "GET",
    headers: headers,
  });
  let data = await response.json();
  return data;
}

async function getProjectWorkflowsData(projectId) {
  let url =
    main_url +
    "/workflows?project_id=" +
    projectId +
    "&skip=0&limit=1000&detailed=false";
  const response = await fetch(url, {
    method: "GET",
    headers: headers,
  });
  let data = await response.json();
  return data;
}

async function getWorkflowData(workflowId) {
  let url = main_url + "/workflows/" + workflowId;
  const response = await fetch(url, {
    method: "GET",
    headers: headers,
  });
  let data = await response.json();
  return data;
}

async function displayOrganisations(node) {
  let selectedWidget = findWidgetByName(node, "organisation");
  let dataWidget = findWidgetByName(node, "data");
  selectedWidget.callback = function () {};
  dataWidget.value = "";

  if (selectedWidget) {
    node.title = "Loading...";
    let data = await getOrganisationsData();
    let selectedOrgs = [];
    if (data && data.length > 0) {
      for (let i = 0; i < data.length; i++) {
        const name = data[i].name;
        selectedOrgs.push(name);
        selectedWidget.options.values.push(name);
      }
      if (selectedOrgs.length > 0) {
        selectedWidget.options.values = selectedOrgs;
        if (selectedOrgs.includes(selectedWidget.value) === false) {
          node.inputs = [];
          node.outputs = [];
          resetWidgets(node);
          selectedWidget.value = selectedOrgs[0];
          await displayProjects(node, data[0]._id);
        } else {
          for (let i = 0; i < data.length; i++) {
            const name = data[i].name;
            if (name === selectedWidget.value) {
              await displayProjects(node, data[i]._id);
              break;
            }
          }
        }
        selectedWidget.callback = async function () {
          for (let i = 0; i < data.length; i++) {
            const name = data[i].name;
            if (name === selectedWidget.value) {
              await displayProjects(node, data[i]._id);
              break;
            }
          }
        };
      } else {
        selectedWidget.value = "No organisation found";
      }
    } else {
      selectedWidget.value = data.detail;
    }
  }
}

async function displayProjects(node, organisationId) {
  let selectedWidget = findWidgetByName(node, "project");
  selectedWidget.callback = function () {};
  let dataWidget = findWidgetByName(node, "data");
  dataWidget.value = "";

  if (selectedWidget) {
    node.title = "Loading...";
    let data = await getProjectsData(organisationId);
    if (data && data.length > 0) {
      let selectedProjects = [];
      for (let i = 0; i < data.length; i++) {
        selectedProjects.push(data[i].name);
      }
      if (selectedProjects.length > 0) {
        selectedWidget.options.values = selectedProjects;
        if (selectedProjects.includes(selectedWidget.value) === false) {
          node.inputs = [];
          node.outputs = [];
          resetWidgets(node);
          selectedWidget.value = selectedProjects[0];
          const projectId = data[0]._id;
          await displayWorkflows(node, projectId, data[0].name);
        } else {
          for (let i = 0; i < data.length; i++) {
            const name = data[i].name;
            if (name === selectedWidget.value) {
              const projectId = data[i]._id;
              await displayWorkflows(node, projectId, name);
              break;
            }
          }
        }
        selectedWidget.callback = async function () {
          for (let i = 0; i < data.length; i++) {
            const name = data[i].name;
            if (name === selectedWidget.value) {
              const projectId = data[i]._id;
              await displayWorkflows(node, projectId, name);
              break;
            }
          }
        };
      } else {
        selectedWidget.value = "No project found";
      }
    } else {
      selectedWidget.value = data.detail;
    }
  }
}

async function displayWorkflows(node, projectId, projectName) {
  let selectedWidget = findWidgetByName(node, "workflow");
  selectedWidget.callback = function () {};
  let dataWidget = findWidgetByName(node, "data");
  dataWidget.value = "";

  if (selectedWidget) {
    node.title = "Loading...";

    let data = await getProjectWorkflowsData(projectId);
    if (data && data.length > 0) {
      let selectedWorkflows = [];
      for (let i = 0; i < data.length; i++) {
        const projects = data[i].projects;
        for (let j = 0; j < projects.length; j++) {
          if (projects[j].name === projectName) {
            selectedWorkflows.push(data[i].name);
            break;
          }
        }
      }

      if (selectedWorkflows.length > 0) {
        selectedWidget.options.values = selectedWorkflows;
        const cached = selectedWorkflows.includes(selectedWidget.value);
        if (cached === false) {
          selectedWidget.value = selectedWorkflows[0];
        }
        for (let i = 0; i < data.length; i++) {
          if (data[i].name === selectedWidget.value) {
            await updateInputsOutputs(node, data[i], true);
            break;
          }
        }

        selectedWidget.callback = async function () {
          for (let i = 0; i < data.length; i++) {
            if (data[i].name === selectedWidget.value) {
              await updateInputsOutputs(node, data[i], true);
              break;
            }
          }
        };
      } else {
        selectedWidget.value = "No workflow found";
      }
    } else {
      selectedWidget.value = data.detail;
    }
  }
}

function resetWidgets(node) {
  let widgets = [];
  widgets.unshift(findWidgetByName(node, "workflow"));
  widgets.unshift(findWidgetByName(node, "project"));
  widgets.unshift(findWidgetByName(node, "organisation"));
  widgets.unshift(findWidgetByName(node, "data"));
  node.widgets = widgets;
}

async function updateInputsOutputs(node, workflowObject, update) {
  if (update) {
    node.outputs = [];
    resetWidgets(node);
  }
  let workflowId = workflowObject.id;
  if (workflowId === undefined) {
    workflowId = workflowObject._id;
  }
  node.title = "Loading...";
  const detailedWorkflow = await getWorkflowData(workflowId);
  const parsedWorkflow = JSON.parse(detailedWorkflow.workflow_api);
  node.title = detailedWorkflow.name;

  let data = {
    origin: main_url,
    token: token,
    workflow_api: parsedWorkflow,
    widget_inputs: [],
  };

  const nodes = Object.keys(parsedWorkflow).map((key) => [key, parsedWorkflow[key]]);
  let newInputs = [];
  for (let i = 0; i < nodes.length; i++) {
    const wfNode = nodes[i][1];
    const nodeType = wfNode.class_type;
    if (nodeType.startsWith("signature_input")) {
      const nodeInputs = wfNode.inputs;
      const required = nodeInputs.required;
      if (required && update) {
        const inputName = nodeInputs.title;
        let inputType = nodeInputs.subtype.toUpperCase();
        const value = nodeInputs.value;
        if (nodeType === "signature_input_text") {
          inputType = "STRING";
        }
        if (inputType !== "IMAGE" && inputType !== "MASK") {
          if (inputType === "INT" || inputType === "FLOAT") {
            let precision = 1;
            let step = 1;
            if (inputType === "FLOAT") {
              precision = 3;
              step = 0.01;
            }
            data.widget_inputs.push({
              title: inputName,
              value: value,
              type: inputType,
            });
            node.addWidget(
              "number",
              inputName,
              value,
              () => {
                const widget = findWidgetByName(node, inputName);
                const newValue = widget.value;
                const updatedData = getData(node);
                for (let i = 0; i < updatedData.widget_inputs.length; i++) {
                  if (updatedData.widget_inputs[i].title === inputName) {
                    updatedData.widget_inputs[i].value = newValue;
                    break;
                  }
                }
                updateData(node, updatedData);
              },
              { precision: precision, step: step },
            );
          } else if (inputType === "BOOLEAN") {
            data.widget_inputs.push({
              title: inputName,
              value: value,
              type: inputType,
            });
            node.addWidget("toggle", inputName, value, () => {
              const widget = findWidgetByName(node, inputName);
              const newValue = widget.value;
              const updatedData = getData(node);
              for (let i = 0; i < updatedData.widget_inputs.length; i++) {
                if (updatedData.widget_inputs[i].title === inputName) {
                  updatedData.widget_inputs[i].value = newValue;
                  break;
                }
              }
              updateData(node, updatedData);
            });
          } else if (inputType === "STRING") {
            data.widget_inputs.push({
              title: inputName,
              value: value,
              type: inputType,
            });
            node.addWidget("text", inputName, value, () => {
              const widget = findWidgetByName(node, inputName);
              const newValue = widget.value;
              const updatedData = getData(node);
              for (let i = 0; i < updatedData.widget_inputs.length; i++) {
                if (updatedData.widget_inputs[i].title === inputName) {
                  updatedData.widget_inputs[i].value = newValue;
                  break;
                }
              }
              updateData(node, updatedData);
            });
          }
        } else {
          if (!node.inputs.find((input) => input.name === inputName)) {
            node.addInput(inputName, inputType);
          }
          newInputs.push(inputName);
        }
      }
    }

    if (nodeType === "signature_output") {
      let nodeOutputs = wfNode.inputs;
      const name = nodeOutputs.title;
      const type = nodeOutputs.subtype.toUpperCase();
      if (update) {
        node.addOutput(name, type);
      }
    }
  }

  const inputsToRemove = node.inputs.filter((input) => !newInputs.includes(input.name));
  for (const input of inputsToRemove) {
    node.removeInput(input.name);
  }
  updateData(node, data);
}

function getData(node) {
  for (let i = 0; i < node.widgets.length; i++) {
    if (node.widgets[i].name === "data") {
      return JSON.parse(node.widgets[i].value);
    }
  }
}

function updateData(node, data) {
  for (let i = 0; i < node.widgets.length; i++) {
    if (node.widgets[i].name === "data") {
      node.widgets[i].value = JSON.stringify(data);
      break;
    }
  }
}

const ext = {
  name: "Signature.Wrapper",
  async nodeCreated(node) {
    const class_name = node.comfyClass;
    if (class_name === "signature_wrapper") {
      let widgets = node.widgets;
      for (let i = 0; i < widgets.length; i++) {
        if (widgets[i].name === "data") {
          widgets[i].type = "tschide";
          break;
        }
      }

      if (widgets.length === 1) {
        widgets.unshift({
          type: "combo",
          name: "workflow",
          value: "loading...",
          options: { values: [] },
        });
        widgets.unshift({
          type: "combo",
          name: "project",
          value: "loading...",
          options: { values: [] },
        });
        widgets.unshift({
          type: "combo",
          name: "organisation",
          value: "loading...",
          options: { values: [] },
        });
        node.outputs = [];
        resetWidgets(node);
      }
      await displayOrganisations(node);
    }
  },
};

app.registerExtension(ext);
