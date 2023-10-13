
if [ $1 = 1 ]; then
    python3 main_2_FL_res.py --env "hopper-medium-v2" --seed 2 --num_agents 10 --device 1
  python3 main_2_FL_res.py --env "hopper-medium-v2" --seed 3 --num_agents 10 --device 1
  python3 main_2_FL_res.py --env "hopper-medium-v2" --seed 4 --num_agents 10 --device 1
  python3 main_2_FL_res.py --env "hopper-medium-v2" --seed 5 --num_agents 10 --device 1
elif [ $1 = 2 ]; then
  python3 main_2_FL_res.py --env "hopper-medium-v2" --seed 3 --num_agents 10 --device 1
  python3 main_2_FL_res.py --env "hopper-medium-v2" --seed 4 --num_agents 10 --device 1
  python3 main_2_FL_res.py --env "hopper-medium-v2" --seed 5 --num_agents 10 --device 1

elif [ $1 = 3 ]; then
  python3 main_2_FL_vae_res.py --env "hopper-medium-v2" --seed 3 --num_agents 10 --device 1
  python3 main_2_FL_vae_res.py --env "hopper-medium-v2" --seed 4 --num_agents 10 --device 1
  python3 main_2_FL_vae_res.py --env "hopper-medium-v2" --seed 5 --num_agents 10 --device 1
else
    echo "无效的选择"
fi
