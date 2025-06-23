# HuggingFace Model Tracing

This extension allows you to trace HuggingFace models directly to generate LLM training data from their execution graphs. You can see exactly how HuggingFace models process inputs internally.

## Features

- **HF Model Tracing**: Trace any HuggingFace model's execution graph
- **Multiple Input Support**: Trace models with different text inputs
- **Automatic Wrapping**: Handles different HF model types (BERT, GPT, etc.)
- **Fallback Tracing**: Component-level analysis when full tracing fails
- **Batch Processing**: Trace multiple models at once

## Installation

Make sure you have the required dependencies:

```bash
pip install transformers torch
```

Or install using the project's dependencies:

```bash
pip install -e .
```

## Quick Start

### Basic Usage

```python
from src.bearer.hf_trace import HuggingFaceModelTracer

# Trace a single HuggingFace model
tracer = HuggingFaceModelTracer("distilbert-base-uncased")
dataset_path = tracer.trace_model("Hello, how are you today?")

# The tracer will:
# 1. Load the HF model
# 2. Tokenize your input text  
# 3. Trace the model's execution graph
# 4. Generate LLM training data about the model's operations
```

### Trace Multiple Models

```python
from src.bearer.hf_trace import trace_popular_hf_models

# Trace several popular models at once
results = trace_popular_hf_models(
    model_names=["distilbert-base-uncased", "gpt2", "distilgpt2"],
    input_text="The quick brown fox jumps over the lazy dog.",
    max_length=64
)
```

### Trace with Multiple Inputs

```python
# Trace the same model with different inputs
tracer = HuggingFaceModelTracer("gpt2")
inputs = [
    "Hello world!",
    "The weather is nice today.", 
    "Machine learning is fascinating.",
    "What time is it?"
]
dataset_path = tracer.trace_multiple_inputs(inputs)
```

## How It Works

1. **Model Loading**: Downloads and loads the specified HuggingFace model
2. **Input Processing**: Tokenizes your input text using the model's tokenizer
3. **Model Wrapping**: Wraps the HF model to make it compatible with PyTorch FX tracing
4. **Graph Tracing**: Uses `torch.fx.symbolic_trace` to capture the model's execution graph
5. **Data Generation**: Creates training data about tensor operations in the model
6. **Fallback Handling**: If full tracing fails, generates basic architectural information

## Example Output

```
🚀 Starting experiment with microsoft/DialoGPT-small
📊 Generating training data from PyTorch model...
[✓] Dataset written to hf_graph_dataset.jsonl
🧪 Testing microsoft/DialoGPT-small on 5 samples...

--- Sample 1 ---
Query: # Model forward pass:

def forward(self, x):
    out = self.linear1(x)
    out = self.relu(out)
    return self.linear2(out)

# Current operation:
Executing: out = self.linear1(x)

# Graph state before this operation:
- x: shape=[1, 4], dtype=torch.float32

What are the shape and dtype after executing this operation?

Ground Truth: Shape: [1, 8], dtype: torch.float32
Prediction: The shape would be [1, 8] and dtype would be torch.float32

📈 Evaluating results...
💾 Results saved to hf_experiment_results_microsoft_DialoGPT-small.json
📊 Shape Accuracy: 80.0%
📊 Dtype Accuracy: 100.0%
```

## Supported Models

The integration works with any HuggingFace text generation model. Some recommended models for testing:

- **Small/Fast**: `microsoft/DialoGPT-small`, `distilgpt2`
- **Medium**: `microsoft/DialoGPT-medium`, `gpt2`
- **Large**: `microsoft/DialoGPT-large`, `gpt2-large`
- **Chat-focused**: `facebook/blenderbot-400M-distill`, `microsoft/DialoGPT-medium`

## Configuration Options

### HuggingFaceGraphTracer Parameters

- `model_name`: HuggingFace model identifier
- `num_test_samples`: Number of samples to test (default: 10)
- `max_length`: Maximum generation length (default: 512)

### Evaluation Metrics

- **Shape Accuracy**: Percentage of samples where the model correctly identifies tensor dimensions
- **Dtype Accuracy**: Percentage of samples where the model correctly identifies tensor data types

## Files Generated

- `hf_graph_dataset.jsonl`: Training data generated from PyTorch model
- `hf_experiment_results_[model_name].json`: Detailed results for single model tests
- `model_comparison_results.json`: Results when comparing multiple models

## Limitations

- Models are tested zero-shot (no fine-tuning)
- Evaluation uses simple heuristics (not perfect semantic matching)
- Large models may be slow to test
- Some models may require specific prompt formatting

## Running the Example

Use the provided example script:

```bash
python example_hf_test.py
```

This will test a CNN model with a HuggingFace language model and show the results. 