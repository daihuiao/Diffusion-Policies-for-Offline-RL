#对比实验4：换一个数据集，看看效果

for my_agent in  3 1
do
  for my_seed in  1 2 3 #seed
  do
    python main_FL.py --env_name "walker2d-medium-replay-v2" --num_agents $my_agent --seed $my_seed
    python main_FL_res.py --env_name "walker2d-medium-replay-v2" --num_agents $my_agent --seed $my_seed
  done
done


