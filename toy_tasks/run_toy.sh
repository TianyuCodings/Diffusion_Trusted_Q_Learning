#!/bin/sh
source activate torch-rl
python toy_main.py --device=3 --env_name=25_gaussian  --reward_type=near  --seed=1 --dir=results_toy_dtql --pretrain_epochs=0 --num_epochs=100 &
python toy_main.py --device=4 --env_name=25_gaussian  --reward_type=far --seed=1 --dir=results_toy_dtql --pretrain_epochs=0 --num_epochs=100 &
python toy_main.py --device=5 --env_name=25_gaussian  --reward_type=hard --seed=1 --dir=results_toy_dtql --pretrain_epochs=0 --num_epochs=100 &

python toy_main.py --device=2 --env_name=swiss_roll_2D  --reward_type=near  --seed=1 --dir=results_toy_dtql --pretrain_epochs=0 --num_epochs=100 &
python toy_main.py --device=1 --env_name=swiss_roll_2D  --reward_type=far --seed=1 --dir=results_toy_dtql --pretrain_epochs=0 --num_epochs=100 &
