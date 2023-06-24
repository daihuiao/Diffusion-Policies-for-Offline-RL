#对比实验2：加残差比不加强
#
#agent1,3，10（选做）
for my_agent in  3 1
do
  for my_seed in  1 2 3 #seed
  do
    python main_FL.py --num_agents $my_agent --seed $my_seed
    python main_FL_res.py --num_agents $my_agent --seed $my_seed
  done
done
python

