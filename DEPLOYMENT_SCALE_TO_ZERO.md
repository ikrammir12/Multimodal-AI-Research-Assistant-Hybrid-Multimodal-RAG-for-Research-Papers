# Low-cost scale-to-zero deployment

This deployment uses:
- Streamlit Community Cloud for the public UI (free Community Cloud deployment).
- Azure Container Apps Consumption plan for the FastAPI backend, with minimum replicas `0` and maximum replicas `1`.

## Important behavior

The UI sends the selected PDF with every question. While a backend replica is alive, the API caches the index in memory. If Azure scales the API to zero and later starts a new replica, the PDF is sent again and the index is rebuilt. No existing `chroma_db/` folder is required. This trades a slower first request after inactivity for avoiding a paid always-on compute instance.

The demo accepts PDFs up to 20 MB and 60 pages. PDFs must contain selectable text; OCR of scanned pages is not implemented. Table detection depends on the PDF layout. Text/table embeddings use BGE-M3, figure search uses CLIP, and answer/figure description uses the OpenAI API.

## Backend environment variables

Configure these in Azure Container Apps > Settings > Secrets / Environment variables:

- `OPENAI_API_KEY`: your OpenAI API key (secret).
- `RAG_API_TOKEN`: a long random secret shared with Streamlit Community Cloud.
- `APP_ENV`: `production`.

Do not commit real secrets to GitHub.

## Local test

From the project root, create `.env` using `.env.example` as a template and fill in the real key/token.

Terminal 1:

```powershell
uv run --with-requirements requirements-api.txt uvicorn api:app --reload --port 8000
```

Terminal 2:

```powershell
uv run --with-requirements requirements.txt streamlit run streamlit_app.py
```

The first upload/query can be slow because BGE-M3 and, when figures are found, CLIP must be downloaded and loaded. The API loads each model only when needed, reuses it for subsequent requests in the same running process, and logs model-load durations and resident memory use. Open `http://127.0.0.1:8501`.

## Model caching and measurements

This version uses the Hugging Face runtime cache rather than bundling model weights into the Docker image:

- `HF_HOME` and `HF_HUB_CACHE` point to `/tmp/huggingface` inside the container.
- BGE-M3 is loaded the first time a PDF is indexed, then reused by later requests in the same running replica.
- CLIP is loaded only when a PDF has embedded figures to index, then reused by later requests.
- Logs report model-load durations, indexing duration, and process resident memory (`process_rss_mb`).
- `/health` reports current process memory, uptime, and whether the models are loaded, without loading models just to answer the health check.

Because `/tmp` is temporary, the model cache may be missing after a replica is replaced or scaled to zero. In that case, weights are downloaded again on the next cold start. This keeps the image smaller but can make cold starts slower. If repeated downloads are too slow, consider a persistent cache or baking pinned model weights into the image later; both have trade-offs.

### Important: request timeout during cold starts

Azure Container Apps HTTP ingress has a 240-second request timeout. The Streamlit client timeout is longer, but it cannot override Azure's ingress timeout. If downloading BGE-M3, optionally downloading CLIP, indexing the PDF, and generating an answer takes more than 240 seconds, that request can fail. Test with a representative PDF and inspect the frontend's end-to-end duration, API logs, `/health` model-load durations, and `process_rss_mb`. If the first request exceeds the limit, use a background indexing job/status flow or bake model weights into the image to avoid runtime model downloads.

## Azure Container Apps backend

Install Azure CLI, then run these commands in PowerShell from the project folder after the code is tested locally:

```powershell
az login
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
```

Choose a resource group and a region available for Container Apps. The following command creates supporting resources as needed, builds the API container, and deploys it. Azure may also create an Azure Container Registry and Log Analytics workspace, which can have separate charges.

```powershell
az containerapp up `
  --name multimodal-rag-api `
  --resource-group rg-multimodal-rag `
  --location centralindia `
  --source . `
  --ingress external `
  --target-port 8000
```

After deployment, note the HTTPS URL printed by the CLI. In Azure Portal, open the Container App and set:

- `OPENAI_API_KEY` as a secret, then reference it as an environment variable.
- `RAG_API_TOKEN` as a secret, then reference it as an environment variable.
- `APP_ENV=production`.
- Minimum replicas: `0`.
- Maximum replicas: `1` (to limit scaling and unexpected compute usage).
- Revision resources: start with a configuration that can accommodate the models; this code's prior guide suggests `4` vCPU and `8 GiB` memory if that size is available. That is not a guaranteed minimum or price recommendation—check the current available configurations and costs in your region first. Monitor `process_rss_mb` in `/health` and model/indexing timings in logs. If it cannot load the models within the configured memory, increase resources only after reviewing the logs and cost estimate.

After saving changes, open `https://YOUR-CONTAINER-APP-URL/health`. The health response should be JSON. Then open `/docs` to see the API endpoints.

## Streamlit Community Cloud frontend

1. Push `streamlit_app.py` and `requirements.txt` to GitHub's `main` branch.
2. Go to https://share.streamlit.io and create an app from your repository with main file `streamlit_app.py`.
3. In Streamlit app settings, add secrets like this, replacing the URL and token:

```toml
RAG_API_URL = "https://YOUR-CONTAINER-APP-URL"
RAG_API_TOKEN = "the_same_long_random_token_as_in_Azure"
```

4. Save, restart the Streamlit app, upload a PDF, and test a question.

## Cost notes

Scale-to-zero avoids Azure Container Apps replica compute charges while no replicas are running, but Azure Container Registry, Log Analytics, image storage/egress, and OpenAI API calls can still incur costs. Since weights are downloaded at runtime, Hugging Face network access and cold-start download time matter too. Configure a cost alert and inspect Cost Analysis. Maximum replicas is set to one for the demo. Do not create a virtual machine or GPU resource for this setup.

## Cleanup

When the demo is no longer needed, delete the Container App and resources created only for it (including the environment, registry, and Log Analytics workspace if unused). Deleting only the app may not delete its supporting resources.
