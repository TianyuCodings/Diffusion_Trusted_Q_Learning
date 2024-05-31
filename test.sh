#!/bin/bash
source activate torch-rl
seed=1
dir="results_v1"
##################################### halfcheetah-medium-v2 ########################################################
python main.py --device=1 --env_name=halfcheetah-medium-v2 --eval_freq=50 --seed=${seed} --dir=${dir} &
#
##################################### halfcheetah-medium-replay-v2 ########################################################
python main.py --device=2 --env_name=halfcheetah-medium-replay-v2 --eval_freq=50 --seed=${seed} --dir=${dir} &

############################################### halfcheetah-medium-expert-v2 #####################################
python main.py --device=3 --env_name=halfcheetah-medium-expert-v2 --eval_freq=50 --seed=${seed} --dir=${dir} &
#
##################################### hopper-medium-v2 ########################################################
python main.py --device=1 --env_name=hopper-medium-v2 --eval_freq=50 --seed=${seed} --dir=${dir} &
#
##################################### hopper-medium-replay-v2 ########################################################
python main.py --device=2 --env_name=hopper-medium-replay-v2 --eval_freq=50 --seed=${seed} --dir=${dir} &
#
################################################ hopper-medium-expert-v2 #####################################
python main.py --device=3 --env_name=hopper-medium-expert-v2 --eval_freq=50 --seed=${seed} --dir=${dir} &

#################################### walker2d-medium-v2 ########################################################
python main.py --device=0 --env_name=walker2d-medium-v2 --eval_freq=50 --seed=${seed} --dir=${dir} &

################################### walker2d-medium-replay-v2 ########################################################
python main.py --device=0 --env_name=walker2d-medium-replay-v2 --eval_freq=50 --seed=${seed} --dir=${dir} &
#
################################################# walker2d-medium-expert-v2 #####################################
#python main.py --device=0 --env_name=walker2d-medium-expert-v2 --eval_freq=50 --seed=${seed} --dir=${dir} &
##
################################################# antmaze-umaze-v0 #####################################
#python main.py --device=1 --env_name=antmaze-umaze-v0 --eval_freq=50 --seed=${seed} --dir=${dir} &
##
################################################# antmaze-umaze-diverse-v0 #####################################
#python main.py --device=2 --env_name=antmaze-umaze-diverse-v0 --eval_freq=50 --seed=${seed} --dir=${dir} &
##
################################################# antmaze-medium-play-v0 #####################################
#python main.py --device=3 --env_name=antmaze-medium-play-v0 --eval_freq=50 --seed=${seed} --dir=${dir} &
##
################################################# antmaze-medium-diverse-v0 #####################################
#python main.py --device=4 --env_name=antmaze-medium-diverse-v0 --eval_freq=50 --seed=${seed} --dir=${dir} &
##
################################################# antmaze-large-play-v0 #####################################
#python main.py --device=5 --env_name=antmaze-large-play-v0 --eval_freq=50 --seed=${seed} --dir=${dir} &
##
################################################# antmaze-large-diverse-v0 #####################################
#python main.py --device=6 --env_name=antmaze-large-diverse-v0 --eval_freq=50 --seed=${seed} --dir=${dir} &
#
################################################ pen-human-v1 #####################################
#python main.py --device=7 --env_name=pen-human-v1 --eval_freq=50 --seed=${seed} --dir=${dir} &

################################################ pen-cloned-v1 #####################################
#python main.py --device=3 --env_name=pen-cloned-v1 --eval_freq=50 --seed=${seed} --dir=${dir} &
#
################################################ kitchen-complete-v0 #####################################
#python main.py --device=4 --env_name=kitchen-complete-v0 --eval_freq=50 --seed=${seed} --dir=${dir} &
#
################################################ kitchen-partial-v0 #####################################
#python main.py --device=5 --env_name=kitchen-partial-v0 --eval_freq=50 --seed=${seed} --dir=${dir} &
#
################################################ kitchen-mixed-v0 #####################################
#python main.py --device=6 --env_name=kitchen-mixed-v0 --eval_freq=50 --seed=${seed} --dir=${dir} &

