# Passage Search UI

Single-page SolidJS app for semantic passage search. No build step required.

## Layout

```
passage-search-ui/
├── index.html              # Shell + import map + entry script
├── css/
│   └── styles.css          # All styles (light/dark)
├── js/
│   ├── main.js             # render() entry
│   ├── config.js           # URL params / constants
│   ├── store.js            # createStore + pane helpers
│   ├── helpers.js          # pure utilities (highlight, bins, CSV…)
│   ├── toast.js            # toast + clipboard + download
│   ├── actions.js          # search, negatives, panes, export, health
│   ├── overlaps.js         # cross-pane document overlap detection
│   └── components/
│       ├── App.js
│       ├── Pane.js
│       ├── SearchForm.js
│       ├── Negatives.js
│       ├── StatusBar.js
│       ├── Histogram.js
│       ├── Results.js
│       ├── HitCard.js
│       └── Highlight.js
└── README.md
```

## Running

Serve the directory over HTTP (modules need a real origin):

```bash
# from the parent of passage-search-ui, or adjust paths
python -m http.server 8080
# then open http://localhost:8080/passage-search-ui/
```

Or let your FastAPI app mount this folder and serve `index.html`.

### Query params

- `?api=https://…` — API base (default: same origin)
- `?panes=2` — enable multi-pane layout (1–6)

### Offline SolidJS

Download the four packages listed in the import map into `./vendor/` and change the map entries to `/ui/vendor/...` (or relative paths).

## Conventions

- Components receive a pane `id` (string) and read the store; they never receive callbacks as props.
- Actions live in `actions.js` and are imported where needed.
- Cross-pane features (`overlaps`) are memos over `store.panes`.
