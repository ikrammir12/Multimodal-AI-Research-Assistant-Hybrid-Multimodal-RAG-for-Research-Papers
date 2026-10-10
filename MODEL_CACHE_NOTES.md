# Model caching: runtime-download version

This version uses Hugging Face's cache at `/tmp/huggingface` inside the running container. It does not download model weights while building the Docker image.

## What happens

1. Azure starts the API container.
2. The first PDF question needs BGE-M3; the backend downloads it if absent, loads it into RAM, and reuses it for later requests in that process.
3. If the PDF contains embedded figures, CLIP is also downloaded on first use and then reused.
4. `/health` reports process uptime, resident memory, and model-load times once a model has loaded. Container logs report request duration and indexing/model timings.
5. When Azure scales to zero, the process and its in-memory models disappear. Temporary cache files may also disappear, so the next cold start might download model weights again.

## How to measure

After deploying, open `/health` before a question, record the `uptime_seconds`, `process_rss_mb`, and model-loaded fields. Upload a representative PDF and ask a question. Then open `/health` again and inspect Azure logs for `model ready in ...s`, `PDF indexing total`, and `HTTP POST /api/ask duration_seconds`. The Streamlit page also displays total request time, which includes the wait for Azure to start the backend.

## Important timeout limitation

Azure Container Apps HTTP ingress has a 240-second request timeout. If the first runtime model download plus indexing/answer generation exceeds that duration, a synchronous upload request may fail. If that occurs, do not simply increase the Streamlit timeout: use an asynchronous job-and-status endpoint, or switch to an image that contains the model files.

## Locally test before cloud deployment

Follow `DEPLOYMENT_SCALE_TO_ZERO.md`. This package has passed Python syntax compilation, but the cloud path and model downloads have not been integration-tested in your Azure subscription.
