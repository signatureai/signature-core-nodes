import { app } from "../../../scripts/app.js";


const ext = {
    name: "signature.rename",

    nodeCreated(node) {
        const title = node.getTitle();
        if (title.startsWith("Signature ")) {
            node.title = title.replace("Signature ", "🔸  ");
        }
    }
};

app.registerExtension(ext);