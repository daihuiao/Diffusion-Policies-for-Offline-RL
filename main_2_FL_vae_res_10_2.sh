#对比实验1：扩散模型rl比bc强
#
#agent1,3，10（选做）

#for my_seed in  1  #seed
#do
#  for env_name in  "halfcheetah-medium-replay-v2" "halfcheetah-medium-expert-v2"  "halfcheetah-medium-v2"# "halfcheetah-expert-v2"
#  do
#      for my_agent in 10 #env_name
#      do
#        python main_2_FL_vae_res.py --num_agents $my_agent --seed $my_seed --env_name $env_name --device 1
#      done
#  done
#done

for seed in 2    #seed
do
  for env_name in "kitchen-partial-v0"  "pen-cloned-v1"  "antmaze-medium-diverse-v0" #"halfcheetah-medium-replay-v2"  "hopper-medium-replay-v2" "walker2d-medium-replay-v2" # "halfcheetah-medium-expert-v2"  "halfcheetah-medium-v2" # "halfcheetah-expert-v2" #env_name
  do
    for num_agents in  10 #num_agents
    do
      python3 main_2_FL_vae_res.py --env $env_name --seed $seed --num_agents $num_agents --device "cuda:1"
    done
  done
done

#对比实验2