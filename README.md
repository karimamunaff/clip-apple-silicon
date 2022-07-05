# clip-apple-silicon
Run CLIP on Apple's M1 GPU. Refer to `clip-silicon-fix.ipynb` and `clip-m1-clean.ipynb` for more details on changes made in order to make CLIP work on Apple Silicon.

# Requirements
1. Python >=3.9
2. Poetry (https://python-poetry.org/docs/#installation)
3. Apple Silcon GPU (M1)
# Setup
```
poetry install
```

# Run
```
poetry run python test_clip_cifar100.py --batch-size=64
```
