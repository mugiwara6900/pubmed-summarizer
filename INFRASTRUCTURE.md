# Running Llama 3.3 70B on the DGX Spark without an OOM

## The trap

The DGX Spark has **128 GB of unified memory, shared by the GPU, the CPU and the OS**. There is no separate VRAM, so a GPU memory overrun can take down the whole machine. CUDA sees about 121.7 GiB of it.

| Checkpoint | Weights on disk | Fits? |
| --- | --- | --- |
| `meta-llama/Llama-3.3-70B-Instruct` (BF16) | about 140 GB | **No.** It is larger than the machine's memory. |
| `nvidia/Llama-3.3-70B-Instruct-FP8` | 72.7 GB | Yes, but with much less headroom |
| `nvidia/Llama-3.3-70B-Instruct-NVFP4` | 42.8 GB | **Yes. Use this one.** |

## The command

Run on the DGX Spark:

```bash
export VLLM_API_KEY="$(openssl rand -hex 32)"   # Share this with the client machine.

docker run -d --name vllm-llama33 \
  --gpus all \
  --ipc=host \
  --restart on-failure:3 \
  -p 192.168.1.100:8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:v0.33.0 \
  nvidia/Llama-3.3-70B-Instruct-NVFP4 \
    --api-key "$VLLM_API_KEY" \
    --served-model-name llama-3.3-70b \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.60 \
    --max-model-len 8192 \
    --max-num-seqs 2 \
    --max-num-batched-tokens 8192
```

Download the weights first, so a slow download isn't mistaken for a failed startup:

```bash
hf download nvidia/Llama-3.3-70B-Instruct-NVFP4
```

## Why each setting

| Setting | Why |
| --- | --- |
| `nvidia/...-NVFP4` | At 4-bit, the weights take about 40 GiB instead of about 130 GiB. This is the main thing that makes the model fit. |
| `--gpu-memory-utilization 0.60` | vLLM claims about 73 GiB: about 40 GiB of weights, with the rest for runtime overhead and KV cache. About 48 GiB of CUDA-visible memory stays free for the OS, Docker and CPU work. |
| `--max-model-len 8192` | The model's default context is far larger. The script needs about 3,300 tokens per request (up to about 3,000 prompt, 128 output), so 8192 leaves margin. |
| `--max-num-seqs 2` | The script sends one request at a time, so 2 is enough. It also limits how much work other LAN clients can run at once. |
| `--max-num-batched-tokens 8192` | The value vLLM's Llama 3.3 recipe recommends. |
| `--ipc=host` | PyTorch uses shared memory, and Docker's default 64 MB limit is too small. |
| `vllm/vllm-openai:v0.33.0` | A pinned release with an ARM64 build, on CUDA 13.0.3, which matches the Spark's 13.0 driver reported on NVIDIA's forum. The Spark's CPU is ARM64, so an x86-only image won't run. |
| `-p 192.168.1.100:8000:8000` | Publishes the API on the LAN address only. |
| `VLLM_API_KEY` | vLLM has no authentication by default, and this API would otherwise be open to the whole LAN. |
| `--restart on-failure:3` | Recovers from a crash, but doesn't reload a 40 GiB model forever if it keeps failing. |

### Why the KV cache can't grow into an OOM

vLLM reserves its KV cache at startup, rather than growing it per request. If a request doesn't fit, vLLM queues or preempts it rather than allocating more. So a bad memory setting should fail when the server starts, not halfway through the run.

The estimate for this model is about 320 KiB per token at BF16 (80 layers × 2 × 8 KV heads × 128 dimensions × 2 bytes). Two full 8192-token requests therefore need about 5 GiB, well inside what's left after the weights.

### How the script helps

- It sends **one request at a time**, so it can't flood the server.
- It caps output at `max_tokens=128` and shortens any abstract over 12,000 characters.
- It **doesn't retry a slow response**, so a busy server doesn't get duplicate work.
- After a read timeout, it disconnects before retrying. vLLM v0.33.0's chat endpoint uses `with_cancellation`, which is designed to stop generation when the client disconnects.

## Check it after startup

Wait for `Application startup complete` in `docker logs -f vllm-llama33`, then run:

```bash
curl -s -H "Authorization: Bearer $VLLM_API_KEY" http://192.168.1.100:8000/v1/models
free -h
```

`free -h` should still show plenty of available memory with the model loaded. If it's under about 16 GiB, lower `--gpu-memory-utilization`. `nvidia-smi` can't help here, because it reports `Memory-Usage: Not Supported` on the Spark's unified memory.

## Expect slow responses

Community reports for this exact model on a DGX Spark are **about 2–3 tokens/s for FP8 and about 4.5–5 tokens/s for NVFP4**. A dense 70B model has to read all its weights for every token, and the Spark's memory bandwidth (about 273 GB/s) sets the ceiling.

At those speeds, a 128-token answer can take about 30–60 seconds. **A 45-second response is normal on this machine, not a fault**, which is why the script's read timeout is 180 seconds.

If throughput matters, a smaller or mixture-of-experts model will run much faster on this hardware.

## What I couldn't verify

- **This command has not been run on a DGX Spark.** I don't have access to one. I checked the checkpoint sizes, image tag, CUDA version and flags against published sources, but the 0.60 setting is my own estimate from the arithmetic above.
- **NVIDIA doesn't list a Llama 3.3 recipe for the Spark.** Its recipe catalog covers other models, so the configuration above comes from vLLM's general Llama 3.3 recipe and Spark guidance.
- **NVFP4 accuracy needs checking.** vLLM's recipe notes that NVFP4 "may require tuning to maintain accuracy on certain tasks". For clinical summaries, compare a sample against FP8. To use FP8, change the checkpoint and raise `--gpu-memory-utilization` to `0.75` (about 91 GiB).

## Sources

- [vLLM recipe: Llama 3.3 70B](https://github.com/vllm-project/recipes/blob/main/Llama/Llama3.3-70B.md)
- [vLLM blog: vLLM on the DGX Spark](https://vllm.ai/blog/2026-06-01-vllm-dgx-spark)
- [NVIDIA playbook: Serve LLMs with vLLM](https://build.nvidia.com/spark/vllm/advanced)
- [NVIDIA forum: Llama 3.3 70B FP8 at 3 tokens/s on a Spark](https://forums.developer.nvidia.com/t/trouble-with-llama-70b-3-3-instruct-fp8-model-at-3-tokens-per-second/360643)
- [NVIDIA forum: Running a full LLM stack on GB10](https://forums.developer.nvidia.com/t/running-a-full-llm-stack-on-dgx-spark-gb10-your-application-litellm-llama-swap-vllm-llama-cpp-ollama/367580)
