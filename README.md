# Experiments/Benchmarks for MLX, Transformers, llama.cpp, llama.cpp via Ollama on Mac

This script was written for MLX and Transformers; llama-bench was used for llama.cpp, ollama run <model\_name> --verbose was used for ollama. 
> Too lazy to do the whole API integration thing

## Usage
`python bench.py --backend <backend> --model-name <hf model name>`
> Only transformers and mlx are supported for now

## Experiment parameters
- **Prompt**: "Do you think it's possible to build a time machine?" 

### llama.cpp
`llama-bench --m <model_name>`

### Ollama
`ollama run <model_name>`

## Variations
- Quantized models
- Batch processing
- Multi user serving - concurrent requests

## To do
- Concurrent models
- More kwargs in argparse
- Memory profiling for transformers
