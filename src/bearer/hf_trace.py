import json
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .trace import trace_and_generate_llm_data

try:
    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print(
        "Warning: transformers library not available. Please install with: pip install transformers"
    )


class HuggingFaceModelTracer:
    """Trace HuggingFace models to generate LLM training data from their execution graphs."""

    def __init__(self, model_name: str, model_type: str = "auto"):
        """
        Initialize with a HuggingFace model for tracing.

        Args:
            model_name: HF model name (e.g., "bert-base-uncased", "gpt2", "distilbert-base-uncased")
            model_type: Type of model to load ("auto", "causal-lm", "sequence-classification")
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required. Install with: pip install transformers"
            )

        self.model_name = model_name
        self.model_type = model_type

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model based on type
        if model_type == "causal-lm":
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        elif model_type == "sequence-classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        else:
            # Try auto first, fallback to others
            try:
                self.model = AutoModel.from_pretrained(model_name)
            except:
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(model_name)
                except:
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        model_name
                    )

        self.model.eval()
        print(f"✅ Loaded {model_name} ({type(self.model).__name__})")

    def create_example_inputs(
        self, text: str = "Hello world, this is a test.", max_length: int = 128
    ) -> Dict[str, torch.Tensor]:
        """
        Create example inputs for the model.

        Args:
            text: Input text to tokenize
            max_length: Maximum sequence length

        Returns:
            Dictionary of input tensors
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )
        return inputs

    def trace_model(
        self,
        input_text: str = "Hello world, this is a test.",
        max_length: int = 128,
        output_path: str = None,
    ) -> str:
        """
        Trace the HuggingFace model and generate LLM training data.

        Args:
            input_text: Text to use as model input
            max_length: Maximum sequence length
            output_path: Where to save the training data

        Returns:
            Path to the generated dataset
        """
        if output_path is None:
            safe_model_name = self.model_name.replace("/", "_").replace("-", "_")
            output_path = f"hf_trace_{safe_model_name}.jsonl"

        print(f"🔍 Tracing {self.model_name}...")
        print(f"📝 Input text: '{input_text}'")

        # Create inputs
        inputs = self.create_example_inputs(input_text, max_length)
        print(f"🔢 Input shape: {inputs['input_ids'].shape}")

        # For tracing, we need to create a wrapper that takes the inputs in the right format
        class ModelWrapper(nn.Module):
            def __init__(self, hf_model):
                super().__init__()
                self.hf_model = hf_model

            def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
                # Use only input_ids for simpler tracing
                outputs = self.hf_model(input_ids)
                if hasattr(outputs, "last_hidden_state"):
                    return outputs.last_hidden_state
                elif hasattr(outputs, "logits"):
                    return outputs.logits
                else:
                    # Return the first tensor from outputs
                    return outputs[0] if isinstance(outputs, tuple) else outputs

        wrapped_model = ModelWrapper(self.model)

        try:
            # Use the existing trace function
            dataset_path = trace_and_generate_llm_data(
                wrapped_model, inputs["input_ids"], output_path
            )

            print(f"✅ Successfully traced {self.model_name}")
            return dataset_path

        except Exception as e:
            print(f"❌ Error tracing {self.model_name}: {e}")

            # Try with a simpler approach - trace just the core model components
            print("🔄 Attempting to trace model components...")
            return self._trace_model_components(inputs, output_path)

    def _trace_model_components(
        self, inputs: Dict[str, torch.Tensor], output_path: str
    ) -> str:
        """
        Fallback method to trace individual model components when full model tracing fails.
        """
        examples = []

        # Get model architecture info
        model_info = {
            "model_name": self.model_name,
            "model_type": type(self.model).__name__,
            "input_shape": list(inputs["input_ids"].shape),
            "vocab_size": self.tokenizer.vocab_size
            if hasattr(self.tokenizer, "vocab_size")
            else "unknown",
        }

        # Try to get some basic layer information
        layer_info = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                layer_info.append(
                    {
                        "name": name,
                        "type": type(module).__name__,
                        "parameters": sum(p.numel() for p in module.parameters()),
                    }
                )

        # Create a basic training example
        prompt = f"""# HuggingFace Model Analysis

Model: {self.model_name}
Type: {model_info['model_type']}
Input shape: {model_info['input_shape']}
Vocabulary size: {model_info['vocab_size']}

# Model Architecture:
{chr(10).join(f"- {layer['name']}: {layer['type']} ({layer['parameters']} params)" for layer in layer_info[:10])}

# Question: What would be the output shape when processing the input tensor with shape {model_info['input_shape']}?"""

        # Try to get actual output for ground truth
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                if hasattr(outputs, "last_hidden_state"):
                    output_shape = list(outputs.last_hidden_state.shape)
                    response = f"Output shape: {output_shape} (hidden states)"
                elif hasattr(outputs, "logits"):
                    output_shape = list(outputs.logits.shape)
                    response = f"Output shape: {output_shape} (logits)"
                else:
                    output_tensor = (
                        outputs[0] if isinstance(outputs, tuple) else outputs
                    )
                    output_shape = list(output_tensor.shape)
                    response = f"Output shape: {output_shape}"
        except Exception as e:
            response = f"Could not determine output shape due to error: {str(e)}"

        example = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
        }
        examples.append(example)

        # Save the data
        with open(output_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        print(f"💾 Saved basic model info to {output_path}")
        return output_path

    def trace_multiple_inputs(
        self, input_texts: List[str], max_length: int = 128, output_path: str = None
    ) -> str:
        """
        Trace the model with multiple different inputs.

        Args:
            input_texts: List of texts to use as inputs
            max_length: Maximum sequence length
            output_path: Where to save the training data

        Returns:
            Path to the generated dataset
        """
        if output_path is None:
            safe_model_name = self.model_name.replace("/", "_").replace("-", "_")
            output_path = f"hf_multi_trace_{safe_model_name}.jsonl"

        all_examples = []

        for i, text in enumerate(input_texts):
            print(f"\n🔍 Tracing input {i+1}/{len(input_texts)}: '{text[:50]}...'")

            try:
                # Create temporary output path
                temp_path = f"temp_trace_{i}.jsonl"
                self.trace_model(text, max_length, temp_path)

                # Read the examples
                with open(temp_path, "r") as f:
                    for line in f:
                        all_examples.append(json.loads(line.strip()))

                # Clean up temp file
                import os

                os.remove(temp_path)

            except Exception as e:
                print(f"❌ Error with input {i+1}: {e}")
                continue

        # Save all examples
        with open(output_path, "w") as f:
            for ex in all_examples:
                f.write(json.dumps(ex) + "\n")

        print(
            f"✅ Traced {len(input_texts)} inputs, generated {len(all_examples)} examples"
        )
        print(f"💾 Saved to {output_path}")
        return output_path


def trace_popular_hf_models(
    model_names: List[str] = None,
    input_text: str = "The quick brown fox jumps over the lazy dog.",
    max_length: int = 64,
) -> Dict[str, str]:
    """
    Trace multiple popular HuggingFace models.

    Args:
        model_names: List of model names to trace
        input_text: Input text for tracing
        max_length: Maximum sequence length

    Returns:
        Dictionary mapping model names to output file paths
    """
    if model_names is None:
        model_names = [
            "distilbert-base-uncased",
            "gpt2",
            "distilgpt2",
            "bert-base-uncased",
        ]

    results = {}

    for model_name in model_names:
        try:
            print(f"\n{'='*60}")
            print(f"🚀 Tracing {model_name}")
            print(f"{'='*60}")

            tracer = HuggingFaceModelTracer(model_name)
            output_path = tracer.trace_model(input_text, max_length)
            results[model_name] = output_path

        except Exception as e:
            print(f"❌ Failed to trace {model_name}: {e}")
            results[model_name] = f"ERROR: {str(e)}"

    # Save summary
    summary_path = "hf_tracing_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"📊 TRACING SUMMARY")
    print(f"{'='*60}")

    for model_name, result in results.items():
        if result.startswith("ERROR"):
            print(f"❌ {model_name}: {result}")
        else:
            print(f"✅ {model_name}: {result}")

    print(f"\n💾 Summary saved to {summary_path}")
    return results


# Example usage
if __name__ == "__main__":
    # Trace a single model
    tracer = HuggingFaceModelTracer("distilbert-base-uncased")
    dataset_path = tracer.trace_model("Hello, how are you today?")

    # Trace multiple models
    # results = trace_popular_hf_models()

    # Trace with multiple inputs
    # inputs = [
    #     "Hello world!",
    #     "The weather is nice today.",
    #     "Machine learning is fascinating.",
    #     "What time is it?"
    # ]
    # multi_path = tracer.trace_multiple_inputs(inputs)
