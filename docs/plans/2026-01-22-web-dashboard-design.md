# Web Dashboard Design

Simple file browser for viewing Shake&Tune results.

## Navigation Structure

Three-level hierarchy:

```
GET /                   → List of all printers (home)
GET /printer/{name}     → Printer detail: Belts & Shaper sections
```

## Pages

### Home (`/`)

Grid of printer cards showing:
- Printer name
- Count of shaper/belt results
- Last activity timestamp

Click a printer to view its results.

### Printer Detail (`/printer/{name}`)

Two sections: "Shaper Results" and "Belt Results"

Each section lists results chronologically (newest first) with:
- Thumbnail preview
- Formatted timestamp
- View link (opens in new tab)
- Download link

## Implementation

### Files

```
service/
├── server.py           # Add routes + helper functions
└── templates/
    ├── home.html       # Printer list
    └── printer.html    # Printer detail with results
```

### New Routes

| Route | Returns |
|-------|---------|
| `GET /` | HTML home page (replaces JSON docs) |
| `GET /printer/{name}` | HTML printer detail |
| `GET /api` | JSON API docs (moved from `/`) |

### Helper Function

```python
def get_all_results() -> dict:
    """Scan results dir, return structured data."""
    # Returns: {"k1v4": {"shaper": [...], "belts": [...]}, ...}
```

### Dependencies

Add `jinja2` to requirements.txt.

## Unchanged Endpoints

These remain exactly as-is (macros depend on them):

- `POST /shaper` → JSON response
- `POST /belts` → JSON response
- `POST /vibrations` → JSON response
- `GET /results/{printer}/{file}` → PNG file
- `GET /latest/{printer}/{type}` → Redirect
- `GET /health` → JSON status
