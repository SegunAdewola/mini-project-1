# MP1

```
mini-project-1/
├── mp1_dry_bean.ipynb      project code
├── report.pdf              project report
├── report/
│   ├── main.pdf            please ignore           
│   ├── refs.bib
│   └── figures/            charts and graph for report
└── cache/                  downloaded data + trained models
```

**Run it**
```bash
jupytext --to ipynb mp1_dry_bean.py   # once, to get the .ipynb
```
Open `mp1_dry_bean.ipynb` and run all cells. The first run downloads the data and trains everything (about 2 minutes on a CPU). Later runs load from `cache/`. Set `FORCE_RETRAIN = True` to retrain.
