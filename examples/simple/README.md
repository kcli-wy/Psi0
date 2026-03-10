 
# Evaluate Policies on SIMPLE
pull `SIMPLE`
```
git submodule update --init --recursive
cd third_party/SIMPLE
```

## Install Dokcer version

Setup environment variables
```
cp .env.sample .env; 
# then edit the .env file to set the "DATA_DIR" to current directory (e.g., `DATA_DIR=/home/***/projects/psi/third_party/SIMPLE/data`)
```

Build docker image
```
docker compose build isaac-sim
```

## Install UV version (TODO)
```
git submodule update --init --recursive third_party/SIMPLE
```

## Evaluation

> REMINDER: Start server before starting the eval client

copy the training dataset to SIMPLE `data`, which will be mounted into the docker container
```
cp -r $DATA_HOME/G1WholebodyBendPick-v0-psi0 ./data/
```

launch the eval client
```
export task=simple/G1WholebodyBendPick-v0
export policy=psi0
export host=localhost
export port=22085
GPUs=1 docker compose run eval $task $policy \
    --host=$host \
    --port=$port  \
    --sim-mode=mujoco_isaac \
    --headless \
    --max-episode-steps=360 \
    --data-format=lerobot \
    --data-dir=data/G1WholebodyBendPick-v0-psi0 \
    --num-episodes=10
```

> After evaluation is done, checkout the videos from `third_party/SIMPLE/data/evals/psi0`