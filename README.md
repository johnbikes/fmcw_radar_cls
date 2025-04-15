# Create venv
- `uv venv --python 3.12`
- `source .venv/bin/activate`
- `uv pip install -r requirements.txt`

# Check data
- `cp .env.template .env`
- edit DATA_DIR in .env
- `python load_data.py`
- should see the output below

```bash
Loaded 5720 examples for label Cars encoded with 0
Loaded 5065 examples for label Drones encoded with 1
Loaded 6700 examples for label People encoded with 2
```