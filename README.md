# URL Verification API

This repo exposes a single FastAPI endpoint that accepts a URL, runs the existing health and page checks, and returns the combined result as JSON.

## Endpoint

- `POST /check`

Example request body:

```json
{
  "url": "https://example.com"
}
```

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

The API will start on `http://localhost:5000`.

Interactive docs are available at `http://localhost:5000/docs`.

## Example curl

```bash
curl -X POST http://localhost:5000/check \
  -H "Content-Type: application/json" \
  -d "{\"url\":\"https://example.com\"}"
```
