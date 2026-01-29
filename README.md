# Video Moment Finder

Semantic video frame search. Paste a YouTube URL, search for moments using text or images.

## Documentation

- [ROADMAP.md](./ROADMAP.md) - Development phases and tasks
- [STATUS.md](./STATUS.md) - Current progress and metrics
- [CLAUDE.md](./CLAUDE.md) - AI assistant context

## Run API (Mock endpoints)

```bash
uv run uvicorn src.api.app:app --reload --port 8000
```

## Run Frontend

```bash
cd frontend && npm run dev
```
