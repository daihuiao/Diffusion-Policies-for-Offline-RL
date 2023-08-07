#对比实验1：扩散模型rl比bc强
#
#agent1,3，10（选做）

for my_seed in  0 1 2 #1  #seed
do
  for env_name in  "halfcheetah-medium-replay-v2" # "halfcheetah-medium-expert-v2"  "halfcheetah-medium-v2"# "halfcheetah-expert-v2"
  do
      python main_2_FL_res_online.py --model_dir "results/FL_res_fixed|agent-10|T-5||ms-offline|k-0|$my_seed|half" --seed $my_seed --env_name $env_name --device 0
  done
done

