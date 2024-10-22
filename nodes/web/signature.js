import { app } from "../scripts/app.js";

const urlParams = new URLSearchParams(window.location.search);
const env = urlParams.get("env");
if (env === undefined) {
  return;
}
let main_url = "https://api.signature.ai/api/v1";
if (env === "staging") {
  main_url = "https://api-staging.signature.ai/api/v1";
}
const workflow_id = urlParams.get("workflow_id");
const url = main_url + `/workflows/${workflow_id}`;
const token = urlParams.get("token");
const headers = getHeaders(token);

function getHeaders(token) {
  const headers = new Headers();
  headers.append("Content-Type", "application/json");
  headers.append("Access-Control-Allow-Origin", "*");
  headers.append("Authorization", `Bearer ${token}`);
  return headers;
}

function showMessage(message, color) {
  app.ui.dialog.show(
    $el("div", [
      $el("p", {
        style: {
          padding: "5px",
          color: color,
          fontSize: "20px",
          maxHeight: "50vh",
          overflow: "auto",
          backgroundColor: "rgba(0,0,0,0.2)",
        },
        textContent: message,
      }),
    ]).outerHTML,
  );
}

async function loadWorkflow(app) {
  try {
    const response = await fetch(url, {
      method: "GET",
      headers: headers,
    });

    // Parse the workflow
    const get_workflow = await response.json();
    const get_workflow_data = JSON.parse(get_workflow["workflow"]);
    console.log(get_workflow);
    console.log(get_workflow_data);
    if (
      get_workflow_data &&
      get_workflow_data.version &&
      get_workflow_data.nodes &&
      get_workflow_data.extra
    ) {
      await app.loadGraphData(get_workflow_data, true, true);
    }
  } catch (error) {
    showMessage(
      "An Error occurerd while loading the workflow from Signature",
      "#ff0000ff",
    );
  }
}

async function saveWorkflow(app) {
  try {
    // Save the workflow
    const workflow = app.graph.serialize();
    const workflow_api = await app.graphToPrompt();

    const data = {
      workflow: JSON.stringify(workflow),
      workflow_api: JSON.stringify(workflow_api["output"]),
    };

    const response = await fetch(url, {
      method: "PUT",
      headers: headers,
      body: JSON.stringify(data),
    });

    if (response.ok) {
      showMessage("Workflow deployed to Signature", "#00ff00ff");
    } else {
      showMessage(
        "An Error occurerd while deploying the workflow to Signature",
        "#ff0000ff",
      );
    }
  } catch (error) {
    showMessage(
      "An Error occurerd while deploying the workflow to Signature",
      "#ff0000ff",
    );
  }
}

function $el(tag, propsOrChildren, children) {
  const split = tag.split(".");
  const element = document.createElement(split.shift());
  if (split.length > 0) {
    element.classList.add(...split);
  }
  if (propsOrChildren) {
    if (typeof propsOrChildren === "string") {
      propsOrChildren = { textContent: propsOrChildren };
    } else if (propsOrChildren instanceof Element) {
      propsOrChildren = [propsOrChildren];
    }
    if (Array.isArray(propsOrChildren)) {
      element.append(...propsOrChildren);
    } else {
      const { parent, $: cb, dataset, style, ...rest } = propsOrChildren;
      if (rest.for) {
        element.setAttribute("for", rest.for);
      }
      if (style) {
        Object.assign(element.style, style);
      }
      if (dataset) {
        Object.assign(element.dataset, dataset);
      }
      Object.assign(element, rest);
      if (children) {
        element.append(...(Array.isArray(children) ? children : [children]));
      }
      if (parent) {
        parent.append(element);
      }
      if (cb) {
        cb(element);
      }
    }
  }
  return element;
}

function deleteElement(name) {
  const element = document.getElementsByClassName(name);
  // Check if any elements were found
  if (element.length > 0) {
    // Loop through each element and remove it from the DOM
    Array.from(element).forEach((el) => el.remove());
  }
}

function getCleanWorkflow() {
  const jsonString = `{
			"last_node_id": 0,
			"last_link_id": 0,
			"nodes": [],
			"links": [],
			"groups": [],
			"config": {},
			"extra": {
			  "ds": {
				"scale": 0.5644739300537778,
				"offset": [
				  581.6344764174625,
				  97.05710697162648
				]
			  }
			},
			"version": 0.4
		  }`;

  return JSON.stringify(JSON.parse(jsonString));
}

function cleanLocalStorage() {
  const cleanWorkflow = getCleanWorkflow();
  const keysToRemove = [];
  // Iterate through all keys in session storage
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);

    // Check if the key is related to workflow data
    if (key.startsWith("Comfy.PreviousWorkflow") || key.startsWith("workflow")) {
      keysToRemove.push(key);
    }
  }

  // Remove the identified keys
  keysToRemove.forEach((key) => {
    localStorage.removeItem(key);
    // localStorage.setItem(key, cleanWorkflow)
  });
}

function cleanSessionStorage() {
  const cleanWorkflow = getCleanWorkflow();
  const keysToRemove = [];
  // Iterate through all keys in session storage
  for (let i = 0; i < sessionStorage.length; i++) {
    const key = sessionStorage.key(i);

    // Check if the key is related to workflow data
    if (key.startsWith("Comfy.PreviousWorkflow") || key.startsWith("workflow")) {
      keysToRemove.push(key);
    }
  }

  // Remove the identified keys
  keysToRemove.forEach((key) => {
    sessionStorage.removeItem(key);
    // localStsessionStorageorage.setItem(key, cleanWorkflow)
  });
}

const ext = {
  // Unique name for the extension
  name: "Signature.Bridge",
  async init(app) {
    cleanLocalStorage();
    cleanSessionStorage();

    deleteElement("comfyui-logo");
    window.addEventListener("message", (event) => {
      console.log(":::::Event", event);
    });

    deleteElement("comfyui-logo");
    window.addEventListener("message", (event) => {
      console.log(":::::Event", event);
    });
  },
  async setup(app) {
    // await instance.loadGraphData(empty_workflow, true, true);
    await loadWorkflow(app);
    if (app.menu) {
      // Ensure the ComfyAppMenu is available
      if (app.bodyTop) {
        const menuItems =
          app.bodyTop.children[0].children[0].children[1].children[0].children[1]
            .children;
        for (let i = 0; i < menuItems.length; i++) {
          const element = menuItems[i];
          console.log(element.ariaLabel);
          if (
            element.ariaLabel === "Save As" ||
            element.innerText === "Browse Templates"
          ) {
            element.parentNode.removeChild(element);
          }
          if (element.ariaLabel === "Save") {
            const link = element.children[0].children[0];
            const icon = link.children[0];
            const label = link.children[1];
            icon.className = "p-menubar-item-icon pi pi-upload";
            label.textContent = "Deploy";

            link.onclick = async function (event) {
              event.preventDefault(); // Prevent default behavior (like form submission)
              event.stopPropagation(); // Stop the event from bubbling up to parent elements
              console.log("save workflow");
              await saveWorkflow(app);
            };
          }
        }
      }
      if (app.bodyLeft) {
        const menuItems = app.bodyLeft.children[0].children;
        for (let i = 0; i < menuItems.length; i++) {
          const element = menuItems[i];
          console.log(element.ariaLabel);
          if (element.ariaLabel === "Workflows") {
            element.parentNode.removeChild(element);
            break;
          }
        }
      }
    }
  },
};

app.registerExtension(ext);
