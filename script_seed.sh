#对比实验1：扩散模型rl比bc强
#
#agent1,3，10（选做）

for my_seed in  1 2 3 #seed
do
  for my_agent in 1 3 10
  do
    python main_FL_bc.py --num_agents $my_agent --seed $my_seed
    python main_FL_res.py --num_agents $my_agent --seed $my_seed
  done
done
python

#对比实验2