# PubMed clinical-outcome summarizer

This script cleans messy PubMed records and summarizes the clinical outcome of each one in a single sentence. It uses Llama 3.3 70B, served by vLLM on a DGX Spark over the local network.

**Video walkthrough:** _YouTube link to be added_

## Files

| File | What it is |
| --- | --- |
| [`pubmed_summarizer.py`](pubmed_summarizer.py) | The deliverable: cleaning, payload, HTTP client and error handling |
| [`INFRASTRUCTURE.md`](INFRASTRUCTURE.md) | The `docker run` command and vLLM arguments that avoid an OOM, with the reasoning and sources |
| [`fake_vllm_server.py`](fake_vllm_server.py) | A stand-in server that can be slow, drop connections or return errors |
| [`test_pubmed_summarizer.py`](test_pubmed_summarizer.py) | 65 tests covering cleaning, payloads, retries and failure modes |

## Run it

```bash
pip install -r requirements.txt
export VLLM_API_KEY=...   # the key the server was started with
python pubmed_summarizer.py                          # runs the sample data from the assignment
python pubmed_summarizer.py --input abstracts.json --output results.jsonl
```

The script writes one JSON line per record, with the status `ok`, `truncated`, `failed`, `rejected` or `skipped`. The exit code is 0 when everything succeeded, 1 when some records failed, 2 for an input or configuration error, and 3 when it stopped because the server looks down.

## Design decisions

- **Missing metadata and missing data are treated differently.** A null title or an author of "Unknown" only adds a warning; the abstract is still summarized. A record with no valid ID or no abstract is set aside, never sent to the model, and still listed in the output with the reasons.
- **Timeouts are sized for this hardware.** A dense 70B model on a DGX Spark has been reported at about 2–5 tokens/s, so a one-sentence answer can take 45 seconds. Connecting is limited to 5 seconds, and reading to 180.
- **Slow responses are not retried; failures are.** Timeouts, dropped connections, and HTTP 408/429/5xx responses get up to three attempts with exponential backoff. A 400 isn't retried, and a 401, 403 or 404 stops the run, because every request would fail the same way.
- **The server is protected.** Requests are sent one at a time, output is capped at 128 tokens, oversized abstracts are shortened (keeping the conclusion), and the run stops after three records in a row fail on connection errors.
- **The prompt guards against invented facts.** The abstract is marked as data, not instructions, and the model is told to use only stated facts and to say when the abstract contradicts itself. Sample record 2 does: it says the null hypothesis was rejected but also that there was no significant improvement.
- **Data stays private.** Abstract text never appears in the logs, proxy settings are ignored so traffic stays on the LAN, and the API key is read from the environment.

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

To watch the 45-second case yourself:

```bash
python fake_vllm_server.py --delay 45
python pubmed_summarizer.py --url http://127.0.0.1:8000/v1/chat/completions
```

## Limitations

- Everything here was tested against the fake server, which returns canned replies. Neither the script nor the Docker command has been run on a real DGX Spark with the real model.
- The `--gpu-memory-utilization 0.60` setting is an estimate. [`INFRASTRUCTURE.md`](INFRASTRUCTURE.md) shows the arithmetic and how to check it after startup.
