#对比实验1：扩散模型rl比bc强
#
#agent1,3，10（选做）

for my_seed in  1  #seed
do
  for my_agent in 10
  do
      for env_name in   "halfcheetah-medium-v2" "halfcheetah-medium-replay-v2"  #"halfcheetah-medium-expert-v2" #env_name
      do
        python main_2_FL_res.py --num_agents $my_agent --seed $my_seed --env_name $env_name --device 1
      done
  done
done
python

#对比实验2