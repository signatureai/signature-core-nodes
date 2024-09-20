import { app } from "../../../scripts/app.js";

const origin = window.location.search;
const urlParams = new URLSearchParams(origin);
const env = urlParams.get("env");
let token = urlParams.get("token");
let main_url = "https://api.signature.ai/api/v1";
if (env === "staging") {
  main_url = "https://api-staging.signature.ai/api/v1";
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

async function getProjectsData(id) {
  let url = main_url + "/projects?organisationId=" + id + "&skip=0&limit=1000";
  const response = await fetch(url, {
    method: "GET",
    headers: headers,
  });
  let data = await response.json();
  return data;
}

async function getWorkflowsData(id) {
  let url = main_url + "/workflows?organisationId=" + id + "&skip=0&limit=1000";
  const response = await fetch(url, {
    method: "GET",
    headers: headers,
  });
  let data = await response.json();
  return data;
}

async function displayOrganisations(node) {
  let widgets = node.widgets;
  let selectedWidget = undefined;
  for (let i = 0; i < widgets.length; i++) {
    if (widgets[i].name === "data") {
      let dataWidget = widgets[i];
      dataWidget.value = "";
    }
    if (widgets[i].name === "organisation") {
      selectedWidget = widgets[i];
      selectedWidget.callback = function () {};
    }
  }

  if (selectedWidget) {
    let data = await getOrganisationsData();
    // console.log("organisation data", data);

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
        // console.log("No organisation found");
      }
    } else {
      selectedWidget.value = data.detail;
      // console.log("No organisation found");
    }
  } else {
    // console.log("No organisation widget found");
  }
}

async function displayProjects(node, id) {
  let widgets = node.widgets;
  let selectedWidget = undefined;

  for (let i = 0; i < widgets.length; i++) {
    if (widgets[i].name === "data") {
      let dataWidget = widgets[i];
      dataWidget.value = "";
    }
    if (widgets[i].name === "project") {
      selectedWidget = widgets[i];
      selectedWidget.callback = function () {};
    }
  }

  if (selectedWidget) {
    let data = await getProjectsData(id);
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
          selectedWidget.value = selectedProjects[0];
          await displayWorkflows(node, id, data[0].name);
        } else {
          for (let i = 0; i < data.length; i++) {
            const name = data[i].name;
            if (name === selectedWidget.value) {
              await displayWorkflows(node, id, name);
              break;
            }
          }
        }
        selectedWidget.callback = async function () {
          for (let i = 0; i < data.length; i++) {
            const name = data[i].name;
            if (name === selectedWidget.value) {
              await displayWorkflows(node, id, name);
              break;
            }
          }
        };
      } else {
        selectedWidget.value = "No project found";
        // console.log("No project found");
      }
    } else {
      selectedWidget.value = data.detail;
      // console.log("No project found");
    }
  } else {
    // console.log("No project widget found");
  }
}

async function displayWorkflows(node, id, projectName) {
  let widgets = node.widgets;
  let selectedWidget = undefined;
  let dataWidget = undefined;

  for (let i = 0; i < widgets.length; i++) {
    if (widgets[i].name === "workflow") {
      selectedWidget = widgets[i];
      selectedWidget.callback = function () {};
    }

    if (widgets[i].name === "data") {
      dataWidget = widgets[i];
      dataWidget.value = "";
    }
  }

  if (selectedWidget) {
    let data = await getWorkflowsData(id);

    if (data && data.length > 0) {
      console.log("workflow data", data);
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
            updateInputsOutputs(node, data[i], cached === false);
            break;
          }
        }

        selectedWidget.callback = async function () {
          for (let i = 0; i < data.length; i++) {
            if (data[i].name === selectedWidget.value) {
              updateInputsOutputs(node, data[i], true);
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

function updateWorkflowImage(node, workflowData) {
  if (workflowData && workflowData.cover_image) {
    const imageWidget = node.widgets.find((w) => w.name === "preview");
    if (imageWidget) {
      imageWidget.value = workflowData.cover_image;
    }
  }
}

function updateInputsOutputs(node, workflowObject, update) {
  if (update) {
    node.inputs = [];
    node.outputs = [];
  }
  const workflowId = workflowObject._id;
  const parsedWorkflow = JSON.parse(workflowObject.workflow_api);
  updateWorkflowImage(node, workflowObject);
  let data = {
    origin: main_url,
    token: token,
    workflow_id: workflowId,
  };

  const nodes = Object.keys(parsedWorkflow).map((key) => [key, parsedWorkflow[key]]);
  for (let i = 0; i < nodes.length; i++) {
    const wfNode = nodes[i][1];
    const nodeType = wfNode.class_type;
    if (nodeType.startsWith("signature_input")) {
      const nodeInputs = wfNode.inputs;
      const required = nodeInputs.required;
      if (required) {
        if (nodeType === "signature_input_connector") {
          const inputName = nodeInputs.title;
          const inputType = "STRING";
          const idName = inputName + " Id";
          const tokenName = inputName + " Token";
          if (update) {
            node.addInput(idName, inputType);
            node.addInput(tokenName, inputType);
          }
        } else {
          const inputName = nodeInputs.title;
          let inputType = nodeInputs.subtype.toUpperCase();
          if (nodeType === "signature_input_text") {
            inputType = "STRING";
          }
          if (update) {
            node.addInput(inputName, inputType);
          }
        }
      }
    }

    if (nodeType === "signature_output") {
      // console.log("output nodes: ", nodes[i]);
      let nodeOutputs = wfNode.inputs;
      const name = nodeOutputs.title;
      const type = nodeOutputs.subtype.toUpperCase();
      if (update) {
        node.addOutput(name, type);
      }
    }
  }

  for (let i = 0; i < node.widgets.length; i++) {
    if (node.widgets[i].name === "data") {
      node.widgets[i].value = JSON.stringify(data);
      break;
    }
  }
}

const ext = {
  // Unique name for the extension
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
      if (widgets.length === 4) {
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

        node.inputs = [];
        node.outputs = [];
      }
      await displayOrganisations(node);
    }
  },
};

app.registerExtension(ext);
