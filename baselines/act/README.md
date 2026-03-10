# ACT
```
uv venv .venv-act --python 3.10
source .venv-act/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync --group psi --group serve --group viz --active --frozen
cp src/lerobot_patch/common/datasets/lerobot_dataset.py \
  .venv-act/lib/python3.10/site-packages/lerobot/common/datasets/lerobot_dataset.py
```