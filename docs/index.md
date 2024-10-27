# Signature Core Nodes Documentation

![GitHub](https://img.shields.io/github/license/signatureai/signature-core-nodes)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?&logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?&logo=opencv&logoColor=white)
![ComfyUI](https://img.shields.io/badge/ComfyUI-compatible-green)

A powerful collection of custom nodes for ComfyUI that provides essential image
processing, data handling, and workflow management capabilities.

📚 **[View Full Documentation](https://signatureai.github.io/signature-core-nodes/)**

## 🚀 Installation

1. Navigate to your ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes/
```

2. Clone the repository:

```bash
git clone https://github.com/signatureai/signature-core-nodes.git ComfyUI/custom_nodes/signature-core-nodes
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 📦 Node Categories

- ⚡ Basic
  - 🧱 Primitives - Basic data type nodes
  - 🔢 Numbers - Numerical operations and processing
  - 🔤 Text - Text processing and manipulation nodes
  - 📁 File - File handling operations
  - 🖼️ Image - Basic image handling nodes
  - 🎭 Mask - Mask generation and operations
- 🖼️ Image Processing - Advanced image processing and manipulation nodes
- 🤖 Models - AI model integration nodes
- 🧠 Logic - Logic operations and control flow
- 🛠️ Utils - Utility functions
- 📦 Others
  - 🔀 Augmentations - Image augmentation tools
  - 🔌 Platform I/O - Platform integration nodes
  - 📊 Data - Data conversion and handling
  - 🧬 Loras - LoRA model handling and integration

## 💻 Usage

After installation, the Signature Core nodes will be available in your ComfyUI workspace
under the "🔲 Signature Nodes" category. Each node is designed to be intuitive and
includes proper input validation and error handling.

### Example Workflow

1. Load an image using `ImageFromWeb` or `ImageFromBase64`
2. Apply transformations using nodes like `ImageTranspose` or `UpscaleImage`
3. Process the image using various filter nodes
4. Export the result using `PlatformOutput` or save directly

## 📁 Project Structure

- `nodes/` - Node implementations
  - `web/` - Web interface components
  - `categories.py` - Node category definitions
  - `shared.py` - Shared utilities and constants
  - `platform_io.py` - Platform integration
  - `wrapper.py` - Workflow wrapper functionality
- `docs/` - Documentation files
- `scripts/` - Development and build scripts

## 🛠 Development Setup

1. Install development dependencies:

```bash
pip install -r dev-requirements.txt
```

2. Install pre-commit hooks:

```bash
pre-commit install
```

The project uses pre-commit hooks for:

- Code formatting and linting
- Syntax checking
- Security checks
- File consistency

3. Generate documentation:

```bash
python scripts/generate_docs.py
```

## 📚 Documentation

Documentation is built using MkDocs with the Material theme. To view the documentation
locally:

1. Install MkDocs and dependencies:

```bash
pip install mkdocs mkdocs-material
```

2. Serve the documentation:

```bash
mkdocs serve
```

The documentation will be available at `http://localhost:8000`.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
