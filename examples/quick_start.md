# Fully test everything on brand new env

0. Install nix first via `sh <(curl --proto '=https' --tlsv1.2 -L https://nixos.org/nix/install) --daemon`
1. `git clone git@github.com:songlin/psi`
2. `git submodule update --init --recursive`
3. pull git lfs for AMO

```bash
cd third_party/SIMPLE
git submodule foreach --recursive 'git lfs install --local || true'
git submodule foreach --recursive 'git lfs pull || true'
```

4. go back to the psi dir, `nix develop` to enter dev shell
5. setup env

```bash
uv venv .venv-psi --python 3.10
source .venv-psi/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync --all-groups --index-strategy unsafe-best-match --active
cp .env.sample .env
```

5. check training: `bash scripts/train/psi0/finetune-simple-psi0.sh G1WholebodyBendPick-v0-psi0 bend-pick`
6. check deployment: `bash scripts/deploy/serve_psi0_simple.sh <ckpt_run_dir> <ckpt_steps>`
7. check eval:

```bash
python examples/simple/simple_eval.py \
 --run-dir <ckpt_run_dir> \
 --ckpt-step 40000 \
 --data-dir <data_dir, e.g. ./data/simple/G1WholebodyBendPick-v0-psi0> \
 --env-id simple/G1WholebodyBendPick-v0 \
 --num-episodes 1 \
 --save-video
```

8. check datagen in SIMPLE:

```bash
cd third_party/SIMPLE
./scripts/tests/check_datagen.sh
```
