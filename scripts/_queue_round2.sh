#!/usr/bin/env bash
cd /home/seung/Workspace/vitacgrasp/projects/graspstress
while pgrep -u seung -f "run_rl_grasp.py" > /dev/null; do sleep 60; done
sleep 20
GS_GPU=0 GS_SEED=44 nohup setsid bash scripts/50_dataset.sh ds_s44 55 > runs/critic/_s44.log 2>&1 < /dev/null &
sleep 5
GS_GPU=1 GS_SEED=45 nohup setsid bash scripts/50_dataset.sh ds_s45 55 > runs/critic/_s45.log 2>&1 < /dev/null &
wait
