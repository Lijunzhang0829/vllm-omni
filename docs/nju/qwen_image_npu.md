# Qwen-Image on NPU

Use the following commands to deploy `Qwen/Qwen-Image` on NPU with the multiprocessing method set to `spawn`, and with VAE slicing and tiling enabled to reduce memory pressure during VAE decoding:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve Qwen/Qwen-Image --omni --port 8099 --vae-use-slicing --vae-use-tiling
```
