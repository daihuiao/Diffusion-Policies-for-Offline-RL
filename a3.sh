#1
#python main_FL_res_no_diffusion.py --num_agents 1  --seed 2
#python main_FL_res_no_diffusion.py --num_agents 1  --seed 3
#python main_FL_res_no_diffusion.py --num_agents 3  --seed 2
#
##2
#python main_FL_res.py --num_agents 1  --seed 2
#python main_FL_res.py --num_agents 1  --seed 3
#python main_FL_res_no_diffusion.py --num_agents 3  --seed 3

#3
python main_FL_res.py --num_agents 3 --env_name "walker2d-medium-replay-v2" --seed 2
python main_FL_res.py --num_agents 3 --env_name "walker2d-medium-replay-v2" --seed 3