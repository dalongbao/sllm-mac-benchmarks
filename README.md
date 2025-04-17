# Experiments/Benchmarks for MLX, Transformers, llama.cpp, llama.cpp via Ollama on Mac

This script was written for MLX and Transformers; llama-bench was used for llama.cpp, ollama run <model\_name> --verbose was used for ollama. 
> Too lazy to do the whole API integration thing

## Usage
`python bench.py --backend <backend> --model-name <hf model name>`
> Only transformers and mlx are supported for now

## Experiment parameters
- **Device**: Macbook Pro 2023 M2 Pro 16GB RAM 19-Core GPU
- **Prompt**: "Do you think it's possible to build a time machine?" (38 tokens for Ollama, 13 tokens for MLX)

### llama.cpp
`llama-bench --m <model\_name>`

### Ollama
`ollama run <model\_name>`

## Metrics
- Time to first token
- Tokens per second 

## Variations
- Quantized models
- Batch processing
- Multi user serving - concurrent requests

## Models tested
- Llama 3.2 3B
- Qwen 2.5 1.5B Instruct
- Gemma 3 4B

## To do
- Concurrent models
- More kwargs in argparse
- Memory profiling for transformers
- conversion of full precision models and run with llama.cpp and ollama
