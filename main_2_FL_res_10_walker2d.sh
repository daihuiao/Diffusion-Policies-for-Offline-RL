#对比实验1：扩散模型rl比bc强
#
#agent1,3，10（选做）

for my_seed in  0 1 2 #1  #seed
do
  for env_name in  "walker2d-medium-replay-v2" # "halfcheetah-medium-expert-v2"  "halfcheetah-medium-v2"# "halfcheetah-expert-v2"
  do
      for my_agent in 10 #env_name
      do
        python main_2_FL_res.py --num_agents $my_agent --seed $my_seed --env_name $env_name --device 1 --save_best_model
      done
  done
done


#对比实验2
## i
#python main_2_FL_res.py --num_agents 10 --seed 0 --env_name "hopper-medium-replay-v2" --device 0
#python main_2_FL_res.py --num_agents 10 --seed 0 --env_name "walker2d-medium-replay-v2" --device 1
#
## g
#python main_2_FL_res.py --num_agents 10 --seed 1 --env_name "hopper-medium-replay-v2" --device 0
#python main_2_FL_res.py --num_agents 10 --seed 1 --env_name "walker2d-medium-replay-v2" --device 1
#
## ps
#python main_2_FL_res.py --num_agents 10 --seed 2 --env_name "hopper-medium-replay-v2" --device 0
#python main_2_FL_res.py --num_agents 10 --seed 2 --env_name "walker2d-medium-replay-v2" --device 1
#python main_2_FL_res.py --num_agents 10 --seed 3 --env_name "hopper-medium-replay-v2" --device 0
#python main_2_FL_res.py --num_agents 10 --seed 3 --env_name "walker2d-medium-replay-v2" --device 1
