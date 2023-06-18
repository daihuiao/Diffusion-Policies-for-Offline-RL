import  wandb, torch, random, numpy as np
import pickle
x_ys = []
import matplotlib.pyplot as plt
import  scienceplots
plt.style.use(['science','ieee'])

# with open("x_y_vae_06-17 22:21.pkl", "rb") as f:
# with open("x_y_MLP_06-17 22:23.pkl", "rb") as f:
# with open("x_y_diffusion_06-17 22:22.pkl", "rb") as f:
with open("x_y_gaussian_tanh_06-17 22:21.pkl", "rb") as f:
    while True:
        try:
            x_y = pickle.load(f)
            x_ys.append(x_y)
        except:
            haha = True
            break
haha = True
def plot_all(x_ys):
    x_y1, x_y2, x_y3 = x_ys[0], x_ys[1], x_ys[2]
    x,y=x_y1[0],x_y1[1]
    plt.scatter(x, y,s=5, label="end-device 1",c="green")
    x, y = x_y2[0], x_y2[1]
    plt.scatter(x, y,s=5, label="end-device 2",c="blue")
    x, y = x_y3[0], x_y3[1]
    plt.scatter(x, y,s=5,label="aggregation server",c="red")
    plt.axis('equal')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
for i in range(10):
    plot_all(x_ys[i])



