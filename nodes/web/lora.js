import { app } from "../../../scripts/app.js";

function handleNumSlots(node, widget) {
  const numSlots = parseInt(widget.value);

  if (!node.widgetsHidden) {
    node.widgetsHidden = [];
  }

  for (let i = 1; i <= 10; i++) {
    const widgets = [...node.widgets, ...node.widgetsHidden].filter((w) =>
      w.name.endsWith(`_${i}`),
    );

    widgets.forEach((w) => {
      if (i <= numSlots) {
        // Show widget - restore from hidden array if it exists
        const hiddenIndex = node.widgetsHidden.indexOf(w);
        const widgetIndex = node.widgets.indexOf(w);
        if (hiddenIndex !== -1 && widgetIndex === -1) {
          node.widgets.push(w);
          node.widgetsHidden.splice(hiddenIndex, 1);
        }
      } else {
        // Hide widget - move to hidden array
        const widgetIndex = node.widgets.indexOf(w);
        if (widgetIndex !== -1) {
          node.widgetsHidden.push(w);
          node.widgets.splice(widgetIndex, 1);
        }
      }
    });
  }

  // Trigger node size recalculation
  handleMode(node, undefined);
}

function handleMode(node, widget) {
  const modeWidget = node.widgets.find((w) => w.name === "mode");
  if (!modeWidget) return;

  if (!node.widgetsHiddenModes) {
    node.widgetsHiddenModes = [];
  }

  const isSimple = modeWidget.value === "Simple";
  const allWidgets = [...node.widgets, ...node.widgetsHiddenModes];

  allWidgets.forEach((w) => {
    const isWeightWidget = w.name.startsWith("weight_");
    const isAdvancedWidget =
      w.name.startsWith("model_weight_") || w.name.startsWith("clip_weight_");

    if (isSimple) {
      if (isWeightWidget) {
        const slotNum = w.name.replace("weight_", "");
        const loraNameWidget = node.widgets.findIndex(
          (widget) => widget.name === `lora_name_${slotNum}`,
        );

        // Only proceed if corresponding lora_name exists
        if (loraNameWidget !== -1) {
          const hiddenIndex = node.widgetsHiddenModes.indexOf(w);
          const widgetIndex = node.widgets.indexOf(w);
          if (hiddenIndex !== -1 && widgetIndex === -1) {
            node.widgets.splice(loraNameWidget + 1, 0, w);
            node.widgetsHiddenModes.splice(hiddenIndex, 1);
          }
        }
      }
      if (isAdvancedWidget) {
        // Hide advanced widgets
        const widgetIndex = node.widgets.indexOf(w);
        if (widgetIndex !== -1) {
          node.widgetsHiddenModes.push(w);
          node.widgets.splice(widgetIndex, 1);
        }
      }
    } else {
      if (isWeightWidget) {
        // Hide simple weight widgets
        const widgetIndex = node.widgets.indexOf(w);
        if (widgetIndex !== -1) {
          node.widgetsHiddenModes.push(w);
          node.widgets.splice(widgetIndex, 1);
        }
      }
      if (isAdvancedWidget) {
        const slotNum = w.name.replace("model_weight_", "").replace("clip_weight_", "");
        const loraNameWidget = node.widgets.findIndex(
          (widget) => widget.name === `lora_name_${slotNum}`,
        );

        // Only proceed if corresponding lora_name exists
        if (loraNameWidget !== -1) {
          const hiddenIndex = node.widgetsHiddenModes.indexOf(w);
          const widgetIndex = node.widgets.indexOf(w);
          if (hiddenIndex !== -1 && widgetIndex === -1) {
            node.widgets.splice(loraNameWidget + 1, 0, w);
            node.widgetsHiddenModes.splice(hiddenIndex, 1);
          }
        }
      }
    }
  });

  // Trigger node size recalculation
  node.setSize(node.computeSize());
}

const nodeWidgetHandlers = {
  num_slots: handleNumSlots,
  mode: handleMode,
};

function widgetLogic(node, widget, isInitial = false) {
  const handler = nodeWidgetHandlers[widget.name];
  if (handler && !isInitial) {
    handler(node, widget);
  }
}

const ext = {
  name: "signature.lora_stacker",

  nodeCreated(node) {
    const className = node.comfyClass;
    if (className === "signature_lora_stacker") {
      // Setup widget value properties first
      for (const w of node.widgets || []) {
        let widgetValue = w.value;
        let originalDescriptor = Object.getOwnPropertyDescriptor(w, "value");
        Object.defineProperty(w, "value", {
          get() {
            let valueToReturn =
              originalDescriptor && originalDescriptor.get
                ? originalDescriptor.get.call(w)
                : widgetValue;
            return valueToReturn;
          },
          set(newVal) {
            const oldVal = widgetValue;

            if (originalDescriptor && originalDescriptor.set) {
              originalDescriptor.set.call(w, newVal);
            } else {
              widgetValue = newVal;
            }

            // Only call widgetLogic if the value actually changed
            if (oldVal !== newVal) {
              widgetLogic(node, w);
            }
          },
        });
      }

      // Trigger initial setup after node is created
      const numSlotsWidget = node.widgets.find((w) => w.name === "num_slots");
      const modeWidget = node.widgets.find((w) => w.name === "mode");

      if (numSlotsWidget) {
        handleNumSlots(node, numSlotsWidget);
      }
      if (modeWidget) {
        handleMode(node, modeWidget);
      }
    }
  },
};

app.registerExtension(ext);

// app.registerExtension({
//     name: "signature.lora_stacker",
//     async beforeRegisterNodeDef(nodeType, nodeData, app) {
//         const class_name = nodeType.comfyClass;
//         if (class_name === "signature_lora_stacker") {
//             // Override the onConnectionsChange method to update visibility
//             nodeType.prototype.onPropertyChanged = function (name, value) {
//                 const oldValue = this[name];
//                 if (oldValue === value) return; // Skip if value hasn't changed

//                 this[name] = value; // Update the value

//                 // Handle num_slots changes
//                 if (name === "num_slots") {
//                     const numSlots = parseInt(value);
//                     // Remove or restore inputs based on selected number of slots
//                     for (let i = 1; i <= 10; i++) {
//                         const inputs = this.inputs.filter(input =>
//                             input.name.endsWith(`_${i}`)
//                         );

//                         if (i > numSlots) {
//                             // Remove inputs for unused slots
//                             inputs.forEach(input => {
//                                 const index = this.inputs.indexOf(input);
//                                 if (index > -1) {
//                                     this.inputs.splice(index, 1);
//                                 }
//                             });
//                         } else {
//                             // Restore inputs if they don't exist
//                             const hasLora = this.inputs.some(input => input.name === `lora_${i}`);
//                             if (!hasLora) {
//                                 this.inputs.push(
//                                     { name: `lora_${i}`, type: '*', link_type: LiteGraph.INPUT },
//                                     { name: `weight_${i}`, type: 'number', default: 1.0 },
//                                     { name: `model_weight_${i}`, type: 'number', default: 1.0 },
//                                     { name: `clip_weight_${i}`, type: 'number', default: 1.0 }
//                                 );
//                             }
//                         }
//                     }
//                 }
//                 // Handle mode changes
//                 else if (name.startsWith("mode_")) {
//                     const slotNum = name.split("_")[1];
//                     const isSimple = value === "Simple";

//                     // Remove or add weight inputs based on mode
//                     if (isSimple) {
//                         // Remove model and clip weights, add simple weight if not exists
//                         this.inputs = this.inputs.filter(input =>
//                             !input.name.match(`(model_weight_${slotNum}|clip_weight_${slotNum})`)
//                         );
//                         if (!this.inputs.some(input => input.name === `weight_${slotNum}`)) {
//                             this.inputs.push({ name: `weight_${slotNum}`, type: 'number', default: 1.0 });
//                         }
//                     } else {
//                         // Remove simple weight, add model and clip weights if not exist
//                         this.inputs = this.inputs.filter(input =>
//                             input.name !== `weight_${slotNum}`
//                         );
//                         if (!this.inputs.some(input => input.name === `model_weight_${slotNum}`)) {
//                             this.inputs.push(
//                                 { name: `model_weight_${slotNum}`, type: 'number', default: 1.0 },
//                                 { name: `clip_weight_${slotNum}`, type: 'number', default: 1.0 }
//                             );
//                         }
//                     }
//                 }
//             };
//         }
//     }
// });
