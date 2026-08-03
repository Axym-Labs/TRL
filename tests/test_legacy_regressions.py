import ast
from pathlib import Path


def test_legacy_backprop_version_two_dispatches_to_the_declared_architecture():
    """The public comparison script must not silently execute its v1 graph for v2."""
    source = Path("comparison/backprop_mnist.py").read_text()
    tree = ast.parse(source)
    model_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MNISTModelReLU"
    )
    forward = next(
        node for node in model_class.body if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    called_attributes = {
        node.func.attr
        for node in ast.walk(forward)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "forward_v1" in called_attributes
    assert "forward_v2" in called_attributes
    assert "forward_v3" in called_attributes
