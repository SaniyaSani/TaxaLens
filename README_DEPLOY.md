# TaxaLens public website deployment package

This package separates the permanent public website from model inference:

- `site/` is a static frontend suitable for GitHub Pages.
- `api.py` is the FastAPI inference service.
- `Dockerfile` packages the API for a GPU host.
- `.github/workflows/pages.yml` deploys the frontend.

## 1. Copy into the repository

Copy the contents of this directory into the repository root so the final paths are:

```
api.py
Dockerfile
.env.example
site/index.html
site/styles.css
site/app.js
site/config.js
.github/workflows/pages.yml
```

Do not copy the enclosing `TaxaLens_web_deployment` folder itself.

## 2. Configure the model API

The API needs the three permanent model bundles:

- Diptera model directory
- Hymenoptera model directory
- order-router directory

Set the variables shown in `.env.example`. The model host must also have the shared DINOv3 and BioCLIP cache.

Run locally or on a GPU host:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
```

Do not start several workers on one small GPU: each worker loads another copy of the models.

## 3. Connect the frontend

Edit `site/config.js`:

```js
window.TAXALENS_CONFIG = {
  apiUrl: "https://your-real-api-domain.example"
};
```

Set the API environment variable to the final GitHub Pages origin:

```
TAXALENS_ALLOWED_ORIGINS=https://saniyasani.github.io
```

## 4. Publish GitHub Pages

In the repository:

1. Open **Settings → Pages**.
2. Under **Build and deployment**, choose **GitHub Actions**.
3. Commit these files to `feature/hymenoptera-teacher`.
4. Open **Actions** and run **Deploy TaxaLens website** if it did not start automatically.

The static site can be published immediately, but identification becomes functional only after `site/config.js` points to a reachable API.

## 5. Important scientific limitation

The present model predicts taxonomy. The “What to inspect next” panel prioritises morphology and applicable published keys, but does not claim that individual anatomical characters were detected automatically. A dedicated character-localisation model is the next scientific phase.

