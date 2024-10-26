import ast
import os
import sys
from pathlib import Path


def extract_classes_with_docs(content: str) -> list[tuple[str, str, dict]]:
    """Extract all classes with their docstrings and input/output types."""
    classes = []
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        docstring = ast.get_docstring(node)
        if not docstring:
            continue

        metadata = _extract_class_metadata(node)
        classes.append((node.name, docstring, metadata))

    return classes


def _extract_class_metadata(node: ast.ClassDef) -> dict:
    """Extract metadata (input types, return types, category) from a class definition."""
    input_types = {}
    return_types = []
    category = None

    for item in node.body:
        if isinstance(item, ast.ClassDef) and item.name == "INPUT_TYPES":
            input_types = extract_input_types(item)
        elif isinstance(item, ast.Assign):
            for target in item.targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id == "RETURN_TYPES":
                    return_types = extract_return_types(item)
                elif target.id == "CATEGORY":
                    category = extract_category(item)

    return {
        "input_types": input_types,
        "return_types": return_types,
        "category": category,
    }


def extract_input_types(node):
    """Extract input types from INPUT_TYPES classmethod."""
    input_types = {}
    try:
        for item in node.body:
            if isinstance(item, ast.Return):
                if isinstance(item.value, ast.Dict):
                    input_types = ast.literal_eval(item.value)
    except (ValueError, SyntaxError, TypeError):
        pass
    return input_types


def extract_return_types(node):
    """Extract return types from RETURN_TYPES assignment."""
    try:
        return ast.literal_eval(node.value)
    except (ValueError, SyntaxError, TypeError):
        return []


def extract_category(node):
    """Extract category from CATEGORY assignment."""
    try:
        if isinstance(node.value, ast.Name):
            return node.value.id
        return None
    except (AttributeError, TypeError):
        return None


def create_category_files(nodes_dir: str, docs_dir: str):
    # Create docs/nodes directory if it doesn't exist
    nodes_docs_dir = os.path.join(docs_dir, "nodes")
    os.makedirs(nodes_docs_dir, exist_ok=True)

    # Process only Python files with documented classes
    for file in os.listdir(nodes_dir):
        if file.endswith(".py") and file != "__init__.py" and file != "categories.py":
            module_name = file[:-3]
            with open(os.path.join(nodes_dir, file)) as f:
                content = f.read()
                classes = extract_classes_with_docs(content)

                # Only create documentation file if there are documented classes
                if classes and any(docstring for _, docstring, _ in classes):
                    doc_file = os.path.join(nodes_docs_dir, f"{module_name}.md")
                    with open(doc_file, "w") as doc:
                        doc.write(f"# {module_name.title()} Nodes\n\n")

                        for class_name, docstring, metadata in classes:
                            if docstring:  # Only document classes with docstrings
                                doc.write(f"## {class_name}\n\n")
                                doc.write(f"{docstring}\n\n")

                                # Document input types
                                if metadata["input_types"]:
                                    doc.write("### Input Types\n\n")
                                    for input_category, inputs in metadata["input_types"].items():
                                        doc.write(f"#### {input_category}\n\n")
                                        for name, type_info in inputs.items():
                                            doc.write(f"- `{name}`: {type_info}\n")
                                    doc.write("\n")

                                # Document return types
                                if metadata["return_types"]:
                                    doc.write("### Return Types\n\n")
                                    for return_type in metadata["return_types"]:
                                        doc.write(f"- `{return_type}`\n")
                                    doc.write("\n")

                                doc.write(f"::: nodes.{module_name}.{class_name}\n\n")


def create_mkdocs_config():
    # Get absolute path to nodes directory
    base_dir = Path(__file__).parent.parent
    nodes_path = str(base_dir)

    return f"""site_name: Signature Nodes Documentation
theme:
  name: material
  features:
    - navigation.instant
    - navigation.tracking
    - navigation.sections
    - navigation.expand
    - navigation.top
    - search.highlight
    - search.share
    - toc.follow
  palette:
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode

plugins:
  - search
  - mkdocstrings:
      default_handler: python
      handlers:
        python:
          paths: ["{nodes_path}"]
          options:
            show_source: true
            show_root_heading: true
            heading_level: 2
            docstring_style: google

markdown_extensions:
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - pymdownx.superfences
  - admonition
  - pymdownx.details
  - pymdownx.emoji:
      emoji_index: !!python/name:materialx.emoji.twemoji
      emoji_generator: !!python/name:materialx.emoji.to_svg

nav:
  - Home: index.md
  - Nodes:
    - Categories: nodes/categories.md
    - Data: nodes/data.md
    - Text: nodes/text.md
    - File: nodes/file.md
    - Image: nodes/image.md
    - Platform IO: nodes/platform_io.md
    - Models: nodes/models.md
    - Lora: nodes/lora.md"""


def copy_readme_to_index(base_dir: Path):
    """Copy README.md to docs/index.md with necessary modifications."""
    readme_path = base_dir / "README.md"
    index_path = base_dir / "docs" / "index.md"

    if not readme_path.exists():
        return

    with open(readme_path, encoding="utf-8") as f:
        content = f.read()

    # Replace the title to match documentation
    content = content.replace("# Signature Core for ComfyUI", "# Signature Core Nodes Documentation")

    # Ensure docs directory exists
    os.makedirs(base_dir / "docs", exist_ok=True)

    # Write the modified content to index.md
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    base_dir = Path(__file__).parent.parent
    nodes_dir = base_dir / "nodes"
    docs_dir = base_dir / "docs"

    # Update the Python path to include the base directory
    # We need to modify this part to ensure mkdocs can find the modules
    sys.path.insert(0, str(base_dir))

    # Create an empty __init__.py in the nodes directory if it doesn't exist
    init_file = nodes_dir / "__init__.py"
    if not init_file.exists():
        init_file.touch()

    os.makedirs(docs_dir, exist_ok=True)

    # Copy README.md to index.md first
    copy_readme_to_index(base_dir)

    # Continue with existing functionality
    create_category_files(str(nodes_dir), str(docs_dir))

    with open(base_dir / "mkdocs.yml", "w") as f:
        f.write(create_mkdocs_config())


if __name__ == "__main__":
    main()
