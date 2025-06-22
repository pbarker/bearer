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
    atok = asttokens.ASTTokens(source, parse=True)
    line_map = {}

    for node in ast.walk(atok.tree):
        if isinstance(node, ast.Assign):
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
                "values": output.detach().cpu().numpy().round(3).tolist(),
            }

        graph_state = {}
        for k, v in env.items():
            if isinstance(v, torch.Tensor):
                graph_state[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}

        lineno = None
        for ln, code in line_map.items():
            if str(node.target) in code:
                lineno = ln
                break

        current_line = (
            source_lines[lineno - 1]
            if lineno and 0 <= lineno - 1 < len(source_lines)
            else "<unknown>"
        )

        prompt = (
            f"# Model forward pass:\n\n"
            f"{source_code}\n\n"
            f"# Current line:\n{current_line.strip()}\n\n"
            f"# Executed node: {node.op}: {node.name}\n\n"
            f"# Graph state:\n"
            + "\n".join(
                f"- {k}: shape={v['shape']}, dtype={v['dtype']}"
                for k, v in graph_state.items()
            )
            + "\n\nWhat is the output of this line?"
        )

        response = f"Output: shape={output_info.get('shape')}, dtype={output_info.get('dtype')}"

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
