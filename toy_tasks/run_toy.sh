#!/bin/sh
source activate torch-rl
python toy_main.py --device=5 --env_name=25_gaussian --distill_loss=diffusion --reward_type=near --actor=sac &
python toy_main.py --device=1 --env_name=25_gaussian --distill_loss=diffusion --reward_type=far --actor=sac  &
python toy_main.py --device=2 --env_name=25_gaussian --distill_loss=diffusion --reward_type=hard --actor=sac --seed=2&


python toy_main.py --device=2 --env_name=25_gaussian --distill_loss=diffusion --reward_type=near --actor=sac --gamma=0.005&
python toy_main.py --device=3 --env_name=25_gaussian --distill_loss=diffusion --reward_type=far --actor=sac  --gamma=0.005&
python toy_main.py --device=6 --env_name=25_gaussian --distill_loss=diffusion --reward_type=hard --actor=sac  --gamma=0.005 --seed=2&

python toy_main.py --device=3 --env_name=25_gaussian --distill_loss=dmd --reward_type=near --actor=implicit  &
python toy_main.py --device=7 --env_name=25_gaussian --distill_loss=dmd --reward_type=far --actor=implicit  --train_epochs=2000&
python toy_main.py --device=1 --env_name=25_gaussian --distill_loss=dmd --reward_type=hard --actor=implicit  --train_epochs=2500&

python toy_main.py --device=2 --env_name=swiss_roll_2D --distill_loss=diffusion --reward_type=near --actor=sac --seed=2 &
python toy_main.py --device=3 --env_name=swiss_roll_2D --distill_loss=diffusion --reward_type=far --actor=sac --eta=5 --seed=2 &

python toy_main.py --device=4 --env_name=swiss_roll_2D --distill_loss=dmd --reward_type=near --actor=implicit --train_epochs=1500&
python toy_main.py --device=1 --env_name=swiss_roll_2D --distill_loss=dmd --reward_type=far --actor=implicit  &
