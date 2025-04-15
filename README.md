# Create venv
- `uv venv --python 3.12`
- `source .venv/bin/activate`
- `uv pip install -r requirements.txt`

# Check data
- `cp .env.template .env`
- edit DATA_DIR in .env
- `python load_data.py`
- you should see the output below, followed by a bar chart, and then a 3x3 grid of distance-doppler matrices

```bash
Loaded 5720 examples for label Cars encoded with 0
Loaded 5065 examples for label Drones encoded with 1
Loaded 6700 examples for label People encoded with 2
```

# Blog reference
- https://blogmljt.netlify.app/posts/radar-doppler/