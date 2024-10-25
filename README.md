# Signature Core for ComfyUI

![GitHub](https://img.shields.io/github/license/signatureai/signature-core-nodes)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?&logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?&logo=opencv&logoColor=white)
![ComfyUI](https://img.shields.io/badge/ComfyUI-compatible-green)

A powerful collection of custom nodes for ComfyUI that provides essential image
processing, data handling, and workflow management capabilities.

## 🌟 Features

- **Image Processing**

  - Advanced image transformations and filters
  - Color manipulation and blending modes
  - Mask generation and operations
  - Background removal and image composition

- **Data Handling**

  - JSON/Dictionary conversions
  - File operations
  - Platform I/O management
  - Primitive type handling

- **Workflow Management**

  - Custom workflow wrapper
  - Platform integration
  - Progress tracking
  - Error handling

- **Augmentations**
  - Random crop
  - Image flipping
  - Composite augmentations
  - Batch processing

## 🚀 Installation

1. Clone this repository into your ComfyUI custom nodes directory:

```bash
git clone https://github.com/yourusername/signature-core-nodes.git ComfyUI/custom_nodes/signature-core-nodes
```

2. Install the required dependencies:

```bash
cd ComfyUI/custom_nodes/signature-core-nodes
pip install -e .
```

## 📦 Node Categories

- 🖼️ Image - Image processing and manipulation nodes
- 🎭 Mask - Mask generation and operations
- 🔤 Text - Text processing and manipulation nodes
- 🔢 Numbers - Numerical operations and processing
- 🔄 Transform - Image transformation tools
- 🧱 Primitives - Basic data type nodes
- 🤖 Models - AI model integration nodes
- 🧠 Logic - Logic operations and control flow
- 📁 File - File handling operations
- 🔀 Augmentation - Image augmentation tools
- 🔌 Platform I/O - Platform integration nodes
- 📊 Data - Data conversion and handling
- 🧬 Loras - LoRA model handling and integration
- 🛠️ Utils - Utility functions

## 💻 Usage

After installation, the Signature Core nodes will be available in your ComfyUI workspace
under the "🔲 Signature Nodes" category. Each node is designed to be intuitive and
includes proper input validation and error handling.

### Example Workflow

1. Load an image using `ImageFromWeb` or `ImageFromBase64`
2. Apply transformations using nodes like `ImageTranspose` or `UpscaleImage`
3. Process the image using various filter nodes
4. Export the result using `PlatformOutput` or save directly

## 🛠 Development

The project is structured with clear separation of concerns:

- `nodes/` - Contains all node implementations
- `web/` - Web interface components and extensions
- `categories.py` - Node category definitions
- `shared.py` - Shared utilities and constants

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
