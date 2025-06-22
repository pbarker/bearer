import ast
import inspect
import json
import textwrap
from typing import Union

import asttokens
import torch
import torch.nn as nn
from torch.fx import symbolic_trace


def resolve(arg, env):
    if isinstance(arg, torch.fx.Node):
        return env[arg.name]
    elif isinstance(arg, (tuple, list)):
        return type(arg)(resolve(a, env) for a in arg)
    elif isinstance(arg, dict):
        return {k: resolve(v, env) for k, v in arg.items()}
    return arg


def extract_forward_source(model):
    source = inspect.getsource(model.__class__.forward)
    return textwrap.dedent(source)


def extract_assign_lines(model):
    source = inspect.getsource(model.__class__.forward)
    source = textwrap.dedent(source)
    atok = asttokens.ASTTokens(source, parse=True)
    line_map = {}

    for node in ast.walk(atok.tree):
        if isinstance(node, (ast.Assign, ast.Return)):
            line_text = atok.get_text(node).strip()
            line_map[node.lineno] = line_text

    return line_map


def trace_and_generate_llm_data(
    model: nn.Module,
    example_input: Union[torch.Tensor, tuple],
    output_path="llm_graph_dataset.jsonl",
):
    model.eval()
    traced = symbolic_trace(model)
    env = {}
    examples = []

    source_code = extract_forward_source(model)
    line_map = extract_assign_lines(model)
    source_lines = source_code.splitlines()

    for i, node in enumerate(traced.graph.nodes):
        entry = {
            "executed_node": f"{node.op}: {node.name}",
            "target": str(node.target),
            "args": str(node.args),
            "kwargs": str(node.kwargs),
        }

        if node.op == "placeholder":
            env[node.name] = (
                example_input
                if not isinstance(example_input, tuple)
                else example_input[0]
            )
        elif node.op == "call_module":
            submod = dict(traced.named_modules())[node.target]
            env[node.name] = submod(resolve(node.args[0], env))
        elif node.op == "call_function":
            env[node.name] = node.target(
                *resolve(node.args, env), **resolve(node.kwargs, env)
            )
        elif node.op == "call_method":
            method = getattr(resolve(node.args[0], env), node.target)
            env[node.name] = method(
                *resolve(node.args[1:], env), **resolve(node.kwargs, env)
            )
        elif node.op == "output":
            env[node.name] = resolve(node.args[0], env)

        output = env.get(node.name, None)
        output_info = {}
        if isinstance(output, torch.Tensor):
            output_info = {
                "shape": list(output.shape),
                "dtype": str(output.dtype),
            }

        graph_state = {}
        for k, v in env.items():
            if isinstance(v, torch.Tensor):
                graph_state[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}

        # Better line mapping logic
        current_line = "<unknown>"
        if node.op == "placeholder":
            current_line = f"# Input parameter: {node.name}"
        elif node.op == "output":
            # Find return statement
            for ln, code in line_map.items():
                if "return" in code:
                    current_line = code
                    break
            if current_line == "<unknown>":
                current_line = "# Return statement"
        else:
            # For call_module, call_function, call_method - find best matching line
            target_str = str(node.target)
            best_match = None
            best_line = None

            for ln, code in line_map.items():
                # Look for the target in the code line
                if target_str in code:
                    # Prefer assignment lines, but also accept return statements
                    if (
                        "=" in code and not code.strip().startswith("#")
                    ) or "return" in code:
                        best_match = code
                        best_line = ln
                        break
                    elif best_match is None:
                        best_match = code
                        best_line = ln

            if best_match:
                current_line = best_match

        # Create more accurate prompt based on node type
        if node.op == "placeholder":
            operation_desc = f"Input tensor '{node.name}' is provided to the model"
            question = f"What are the shape and dtype of input '{node.name}'?"
        elif node.op == "output":
            operation_desc = f"Final output is returned: {current_line.strip()}"
            question = "What are the shape and dtype of the final output?"
        else:
            operation_desc = f"Executing: {current_line.strip()}"
            question = f"What are the shape and dtype after executing this operation?"

        prompt = (
            f"# Model forward pass:\n\n"
            f"{source_code}\n\n"
            f"# Current operation:\n{operation_desc}\n\n"
            f"# Graph state before this operation:\n"
            + "\n".join(
                f"- {k}: shape={v['shape']}, dtype={v['dtype']}"
                for k, v in graph_state.items()
                if k
                != node.name  # Don't include current node's output in "before" state
            )
            + f"\n\n{question}"
        )

        # Create more descriptive response
        if output_info:
            response = f"Shape: {output_info['shape']}, dtype: {output_info['dtype']}"
        else:
            response = "No tensor output (this operation doesn't produce a tensor)"

        examples.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]
            }
        )

    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"[✓] Dataset written to {output_path}")
    return output_path


# Example usage
if __name__ == "__main__":

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(4, 8)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(8, 1)

        def forward(self, x):
            out = self.linear1(x)
            out = self.relu(out)
            return self.linear2(out)

    model = Net()
    input_tensor = torch.randn(1, 4)
    trace_and_generate_llm_data(
        model, input_tensor, output_path="llm_graph_dataset.jsonl"
    )
