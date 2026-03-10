## OpenPI Baseline for H-VLA

### Download pretrained `pi05_droid`

```
python scripts/data/download.py \
--repo-id=songlinwei/hfm-models \
--remote-dir=openpi/pi05_droid.nebula101 \
--repo-type=model \
--local-dir=/data1/hfm/cache/checkpoints/
```

### Setup using the original pi repo to obtain pytorch `pi05_droid` checkpoint

```
uv venv .venv-openpi --python 3.10
source .venv-openpi/bin/activate
VIRTUAL_ENV=.venv-openpi uv pip install -e .
cd src/openpi
VIRTUAL_ENV=.venv-openpi GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cd ../../
VIRTUAL_ENV=.venv-openpi GIT_LFS_SKIP_SMUDGE=1 uv pip install -r src/openpi/requirements-openpi.txt
```


Hack `transformers`
```
cp -r  src/openpi/models_pytorch/transformers_replace/* .venv-openpi/lib/python3.10/site-packages/transformers/
```

Download `pi05-droid` checkpoints

```
python src/openpi/shared/download.py
```

Convet from `jax` to `pytorch` checkpoint
```
uv run examples/convert_jax_model_to_pytorch.py \
	--checkpoint_dir=/home/user/.cache/openpi/openpi-assets/checkpoints/pi05_droid \
	--config_name=pi05_droid \
	--output_path /home/user/.cache/openpi/openpi/pytorch_checkpoints/pi05_droid
```

### Start training
modify config.py
```
vim src/openpi/training/config.py
# reference the lerobot data format 
# /path/to/data/Pick_toys_into_box_and_lift_and_turn_and_put_on_the_chair_new_target_yaw/meta/info.json
# to edit inputs & outputs of 
vim src/openpi/training/config.py
```

compute stats (or copy from `src/assets`)
```
python src/openpi/compute_norm_stats.py --config-name Pick_toys_into_box_and_lift_and_turn_and_put_on_the_chair_new_target_yaw
```
or
```
cp src/we/assets/dataset_statistics/g1-stats-put-toys-box-lift-put-chair_target_yaw.json \
/hfm/data/Pick_toys_into_box_and_lift_and_turn_and_put_on_the_chair_new_target_yaw/norm_stats.json
```
or
```
python src/openpi/rewrite_norm_stats.py \
	--task_path=/hfm/data/simple/G1WholebodyBendPick-v0-psi0
```

launch training 
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nnodes=1 --nproc_per_node=8 src/openpi/train_pytorch.py \
	simple_bend_pick \
	--exp_name=simple_bend_pick \
	--save_interval=5000 \
	--checkpoint_base_dir=/hfm/cache/checkpoints/pi05_droid
```

### More implementation details

1. change action dim from `32` to `36`
	```
	# vim src/openpi/models_pytorch/pi0_pytorch.py
	self.action_in_proj = nn.Linear(36, action_expert_config.width)
	self.action_out_proj = nn.Linear(action_expert_config.width, 36)
	```

2. adapt the loading of pretrained `pi05_torch`
	```
	model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
	# Psi-0: adapt action dim to 36
	from safetensors.torch import load_file
	state_dict = load_file(model_path)
	pad_dim = config.model.action_dim - state_dict["action_in_proj.weight"].shape[1]
	if pad_dim > 0:
		# eg., torch.Size([1024, 32]) -> torch.Size([1024, 36])
		# Replicate the last 4 columns instead of padding with zeros
		w = state_dict["action_in_proj.weight"]
		to_pad = w[:, -pad_dim:]
		state_dict["action_in_proj.weight"] = torch.cat([w, to_pad], dim=1)

		b = state_dict["action_out_proj.bias"]
		state_dict["action_out_proj.bias"] = torch.cat([b, b[-pad_dim:]], dim=0)

		w = state_dict["action_out_proj.weight"]
		to_pad = w[-pad_dim:, :]
		state_dict["action_out_proj.weight"] = torch.cat([w, to_pad], dim=0)

	# https://github.com/Physical-Intelligence/openpi/issues/669
	state_dict["paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"] = \
		state_dict["paligemma_with_expert.paligemma.lm_head.weight"]

	_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
	missing_keys, unexpected_keys = _model.load_state_dict(state_dict, strict=False)
	```

3. config.py
	```
	TrainConfig(
        name="Pick_toys_into_box_and_lift_and_turn_and_put_on_the_chair_new_target_yaw",
        project_name="hfm",
        num_workers=8,
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=36,
            action_horizon=16,
            max_token_len=250,
        ),
        data=LeRobotHFMDataConfig( # FIXME
            repo_id= f"{os.environ['DATA_HOME']}/Pick_toys_into_box_and_lift_and_turn_and_put_on_the_chair_new_target_yaw",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=40_000,
        batch_size=128,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1e-4,
            decay_steps=40_000,
            decay_lr=1e-8,
        ),
        pytorch_weight_path=os.environ["PYTORCH_WEIGHT_PATH"],
        policy_metadata={"dataset": "Pick_toys_into_box_and_lift_and_turn_and_put_on_the_chair_new_target_yaw"},
    ),
	```

4. launch training
```
bash scripts/train/openpi/benchmark_pi05_nv_slurm.sh \
	Pick_toys_into_box_and_lift_and_turn_and_put_on_the_chair_new_target_yaw
```

### upload model weights

```
#export task=Hold_lunch_bag_with_both_hands_and_squat_to_put_on_the_coffee_table
#export task=Pick_toys_into_box_and_lift_and_turn_and_put_on_the_chair_new_target_yaw
export task=Pull_the_tray_out_of_chips_can_and_throw_the_can_into_trash_bin
export step=40000
hf upload songlinwei/hfm-models \
	.runs/openpi-05/$task/$task/$step/model.safetensors \
	benchmarks/openpi-05/$task/$step/model.safetensors \
	--repo-type=model
```


### Serve
Download:
```
export task=Remove_the_cap_turn_on_the_faucet_and_fill_the_bottle_with_water
export step=40000
python scripts/data/download.py \
	--repo-id=songlinwei/hfm-models \
	--remote-dir=benchmarks/openpi-05/$task/$step \
	--repo-type=model \
	--local-dir=.runs/openpi-05/$task/$task/$step
```
and Serve:
```
export port=9000
bash scripts/deploy/serve_pi05.sh $task $step $port
```
Open-loop evaluation
```
export PYTORCH_WEIGHT_PATH=/hfm/songlin/we_learn/.cache/checkpoints/pi05_droid
export DATA_HOME=/hfm/songlin/we_learn/.data/real_teleop_g1/lerobot
export task=Pick_bottle_and_turn_and_pour_into_cup
export port=9000

python scripts/train/openpi/eval_openloop.py --port=$port --task=$task
```