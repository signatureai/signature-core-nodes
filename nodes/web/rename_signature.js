import { app } from "../../../scripts/app.js";


const ext = {
    name: "signature.rename",

    nodeCreated(node) {
        const title = node.getTitle();
        if (title.startsWith("SIG ")) {
            node.title = title.replace("SIG ", "🔸  ");
        }
    }
};

app.registerExtension(ext);