# Langfuse Prompt Assets

Use these files to create a Langfuse prompt with path:

`autologin/login-portal-classifier`

Recommended label:

`stage2`

Files:

- `login_portal_classifier.system.txt`
- `login_portal_classifier.user.txt`
- `login_portal_classifier.config.json`

Prompt variables expected by `app.py`:

- `url`
- `soft_errors`
- `visible_text`
- `page_metadata`

Environment variables needed:

```env
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=http://langfuse.diro.live:3000
LANGFUSE_PROMPT_PATH=autologin/login-portal-classifier
LITELLM_PROXY_URL=https://m90-llm-proxy-236951099948.europe-west3.run.app
```

Note:

- If your existing `.env` uses `LITELLM_PROXY`, `app.py` now aliases it to `LITELLM_PROXY_URL`.
- `app.py` also defaults to `autologin/login-portal-classifier` if `LANGFUSE_PROMPT_PATH` is not set.
