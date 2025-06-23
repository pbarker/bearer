#!/usr/bin/env python3
"""
Example script to demonstrate HuggingFace model tracing.

This script shows how to:
1. Trace a single HuggingFace model
2. Trace multiple models
3. Use different input texts
"""

from src.bearer.hf_trace import HuggingFaceModelTracer, trace_popular_hf_models


def trace_single_model():
    """Trace a single HuggingFace model."""
    print("🔍 Tracing a single HuggingFace model...")

    # Create tracer for DistilBERT
    tracer = HuggingFaceModelTracer("distilbert-base-uncased")

    # Trace with a simple input
    dataset_path = tracer.trace_model(
        input_text="The weather is beautiful today.", max_length=64
    )

    print(f"✅ Tracing complete! Data saved to: {dataset_path}")


def trace_multiple_models():
    """Trace multiple popular HuggingFace models."""
    print("\n🔄 Tracing multiple HuggingFace models...")

    models_to_trace = [
        "distilbert-base-uncased",  # Small BERT model
        "distilgpt2",  # Small GPT model
        # "gpt2",                   # Uncomment for larger GPT model
    ]

    results = trace_popular_hf_models(
        model_names=models_to_trace,
        input_text="Machine learning is revolutionizing technology.",
        max_length=64,
    )

    print(
        f"✅ Traced {len([r for r in results.values() if not r.startswith('ERROR')])} models successfully!"
    )


def trace_with_multiple_inputs():
    """Trace a model with multiple different inputs."""
    print("\n📝 Tracing with multiple inputs...")

    tracer = HuggingFaceModelTracer("distilgpt2")

    input_texts = [
        "Hello, how are you?",
        "The cat sat on the mat.",
        "Artificial intelligence is fascinating.",
        "What a beautiful sunset!",
        "Programming in Python is fun.",
    ]

    dataset_path = tracer.trace_multiple_inputs(input_texts, max_length=32)

    print(f"✅ Multi-input tracing complete! Data saved to: {dataset_path}")


def main():
    """Run the examples."""
    print("🚀 HuggingFace Model Tracing Examples\n")

    try:
        # Example 1: Single model
        trace_single_model()

        # Example 2: Multiple models
        trace_multiple_models()

        # Example 3: Multiple inputs
        trace_with_multiple_inputs()

        print("\n" + "=" * 60)
        print("🎉 All examples completed successfully!")
        print("Check the generated .jsonl files to see the tracing data.")

    except Exception as e:
        print(f"❌ Error during tracing: {e}")
        print("Make sure you have transformers installed: pip install transformers")


if __name__ == "__main__":
    main()
