#对比实验1：扩散模型rl比bc强
#
#agent1,3，10（选做）

#for my_seed in  0 1 2 3 4 5 6 7 8 9 #1  #seed
#do
#  for env_name in  "walker2d-medium-replay-v2"  "halfcheetah-medium-replay-v2"  "hopper-medium-replay-v2" #"halfcheetah-medium-v2"# "halfcheetah-expert-v2"
#  do
#      for my_agent in 10
#      do
#        python main_2_FL_res.py --num_agents $my_agent --seed $my_seed --env_name $env_name --device 1 --save_best_model
#      done
#  done
#done

for seed in 2    #seed
do
  for env_name in "kitchen-partial-v0"  "pen-cloned-v1"  "antmaze-medium-diverse-v0" #"halfcheetah-medium-replay-v2"  "hopper-medium-replay-v2" "walker2d-medium-replay-v2" # "halfcheetah-medium-expert-v2"  "halfcheetah-medium-v2" # "halfcheetah-expert-v2" #env_name
  do
    for num_agents in  10 #num_agents
    do
      python3 main_2_FL_res.py --env $env_name --seed $seed --num_agents $num_agents --device 1
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
