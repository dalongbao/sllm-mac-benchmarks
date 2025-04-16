import time
import sys
import argparse
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from mlx_lm import load, generate

def main(args):
    backend = args.backend
    model_name = args.model_name
    quantization = args.quantization
    max_tokens = args.max_tokens
    prompt=args.prompt

    if backend == "mlx":
        mlx_bench(model_name, quantization, max_tokens, prompt)
    elif backend == "transformers":
        transformers_bench(model_name, quantization, max_tokens, prompt)

def mlx_bench(
    model_name, 
    quantization, 
    max_tokens,
    prompt
):
    start = time.time()

    model, tokenizer = load(model_name)
    loadtime = time.time()

    text = generate(
        model, 
        tokenizer, 
        prompt=prompt, 
        max_tokens=max_tokens,
        verbose=True
    )
    generated = time.time()

    print(f"Model: {model_name}")
    print(f"Loading time: {loadtime - start}")
    print(f"Generation time: {generated - loadtime}")


def transformers_bench(
    model_name, 
    quantization, 
    max_tokens,
    prompt
):
    start = time.time()

    model = AutoModelForCausalLM.from_pretrained(model_name).to("mps")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    load = time.time()

    model_inputs = tokenizer([prompt], return_tensors="pt").to("mps")
    tokenized = time.time()

    generated_ids = model.generate(**model_inputs, max_length=max_tokens)
    tokens = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    generated = time.time()

    print(f"User: {prompt}")
    print(f"{model_name}: {tokens}")

    print(f"Model: {model_name}")
    print(f"Loading time: {load - start:.4f} seconds")
    print(f"Tokenizing time: {tokenized - load:.4f} seconds")
    print(f"Generation time: {generated - tokenized:.4f} seconds")
    print(f"Number of generated tokens: {len(generated_ids[0])}")
    print(len(model_inputs["input_ids"][0]))
    print(f"Tokens per second (tokenization): {len(model_inputs['input_ids'][0]) / (tokenized - load):.2f}")
    print(f"Tokens per second (eval): {len(generated_ids[0]) / (generated - tokenized):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference backend benchmark script") 
    parser.add_argument(
        "--backend",
        type=str,
        help=(
            "The backend you want to test. Valid options are mlx, transformers, llama.cpp, and ollama (Ollama separately from llama.cpp because of scheduling differences)."
        ),
        choices=["mlx", "transformers", "llama.cpp", "ollama"],
        required=True,
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help=(
            "The name of the model you want to test, as provided by HuggingFace"
        ),
        required=True,
    )
    parser.add_argument(
        "--quantization",
        type=str,
        help=(
            "Precision at which you want the model to be run. Available precisions are int8, fp4, and nf4"
        ),
        required=False,
        default=None,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help=(
            "Maximum number of tokens generated. Defaults to 128."
        ),
        default=128
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help=(
            "Prompt used to query the model. Defaults to \"Do you think it's possible to build a time machine?\""
        ),
        default="Do you think it's possible to build a time machine?"
    )

    args = parser.parse_args()
    main(args)
