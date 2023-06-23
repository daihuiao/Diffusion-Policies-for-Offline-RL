import wandb, torch, random, numpy as np
import pickle

import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'ieee'])

# # 定义均值和协方差矩阵
# mean0 = [0.5, 0]
# cov0 = [[0.01, 0], [0, 0.01]]
# mean1 = [-0.5, 0]
# cov1 = [[0.01, 0], [0, 0.01]]
# # 生成二维高斯分布
# x0, y0 = np.random.multivariate_normal(mean0, cov0, 50000).T
# x1, y1 = np.random.multivariate_normal(mean1, cov1, 50000).T
# # 绘制二维高斯分布的散点图
# # plt.scatter(x0, y0,s=1)
# plt.scatter(x0[:500], y0[:500],color=(130/255, 178/255, 154/255),label=r"$\mathbf{s}_1$", s=5)
# # plt.scatter(x1, y1,s=1)
# plt.scatter(x1[:500], y1[:500], color=(242/255, 204/255, 142/255),label=r"$\mathbf{s}_2$",s=5)
# plt.scatter(np.concatenate((x0[500:750], x1[750:1000])),
#             np.concatenate((y0[500:750], y1[750:1000])),
#             color=(223/255, 122/255, 94/255),label=r"$\mathbf{s}_0$",s=5)
#             # color=(242/255, 204/255, 142/255),s=5)
#             # color = (60 / 255, 40 / 255, 91 / 255), s = 5)
# plt.axis('equal')
# plt.legend(frameon=True, loc='upper right',)
# plt.show()

x_ys= []
x_ys_all = []





# with open("x_y_vae_06-17 22:21.pkl", "rb") as f:
with open("x_y_MLP_06-17 22:23.pkl", "rb") as f:
# with open("x_y_diffusion_06-17 22:22.pkl", "rb") as f:
# with open("x_y_gaussian_tanh_06-17 22:21.pkl", "rb") as f:
    while True:
        try:
            x_y = pickle.load(f)
            x_ys.append(x_y)
        except:
            haha = True
            break
haha = True
x_ys_all.append(x_ys[5])
x_ys= []

with open("x_y_vae_06-17 22:21.pkl", "rb") as f:
# with open("x_y_MLP_06-17 22:23.pkl", "rb") as f:
# with open("x_y_diffusion_06-17 22:22.pkl", "rb") as f:
# with open("x_y_gaussian_tanh_06-17 22:21.pkl", "rb") as f:
    while True:
        try:
            x_y = pickle.load(f)
            x_ys.append(x_y)
        except:
            haha = True
            break
haha = True
x_ys_all.append(x_ys[9])
x_ys= []

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
x_ys_all.append(x_ys[8])
x_ys= []

# with open("x_y_vae_06-17 22:21.pkl", "rb") as f:
# with open("x_y_MLP_06-17 22:23.pkl", "rb") as f:
with open("x_y_diffusion_06-17 22:22.pkl", "rb") as f:
# with open("x_y_gaussian_tanh_06-17 22:21.pkl", "rb") as f:
    while True:
        try:
            x_y = pickle.load(f)
            x_ys.append(x_y)
        except:
            haha = True
            break
haha = True
x_ys_all.append(x_ys[6])
x_ys= []
def plot_all(x_ys,axs,i,j):
    x_y1, x_y2, x_y3 = x_ys[0], x_ys[1], x_ys[2]
    x, y = x_y1[0], x_y1[1]
    axs[i,j].scatter(x, y, s=5, color=(130/255, 178/255, 154/255),label=r"$\mathbf{s}_1$")
    x, y = x_y2[0], x_y2[1]
    axs[i, j].scatter(x, y, s=5, color=(242/255, 204/255, 142/255),label=r"$\mathbf{s}_2$")
    x, y = x_y3[0], x_y3[1]
    axs[i, j].scatter(x, y, s=5, color=(223/255, 122/255, 94/255),label=r"$\mathbf{s}_0$")
    axs[i, j].axis('equal')
    axs[i, j].legend(frameon=True, loc='upper right', )
    axs[i, j].set_xlabel("x")
    axs[i, j].set_ylabel("y")
    # axs[i, j].set_title('Sin(x)')
    # axs[i, j].show()

# plot_all(x_ys[6])

# for i in range(10):
#     plot_all(x_ys[i])
fig, axs = plt.subplots(2, 2, figsize=(8, 6))

plot_all(x_ys_all[0], axs, 0, 0)
axs[0,0].set_title('MLP')
plot_all(x_ys_all[1], axs, 0, 1)
axs[0,1].set_title('VAE')
plot_all(x_ys_all[2], axs, 1, 0)
axs[1,0].set_title('gaussian')
plot_all(x_ys_all[3], axs, 1, 1)
axs[1,1].set_title('diffusion')

# 设置子图间的距离和整体标题
plt.subplots_adjust(wspace=0.3, hspace=0.3, top=0.9)
# plt.suptitle('comparison of distribution reconstruction effect of four models')

# 显示图形
plt.show()