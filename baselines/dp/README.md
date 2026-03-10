# Diffusion Policy
```
uv venv .venv-dp --python 3.10
source .venv-dp/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync --group serve --group viz --active --frozen
VIRTUAL_ENV=.venv-dp uv pip install -e .
VIRTUAL_ENV=.venv-dp uv pip install -r baselines/dp/requirements-dp.txt
cp src/lerobot_patch/common/datasets/lerobot_dataset.py \
  .venv-dp/lib/python3.10/site-packages/lerobot/common/datasets/lerobot_dataset.py
```