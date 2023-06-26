import pandas as pd
import matplotlib.pyplot as plt
import  numpy as np
import  scienceplots
plt.style.use(['science','ieee'])

# 读取CSV文件并转换为Pandas DataFrame
data = pd.read_csv('agent1_diffusion_effect.csv')
x=np.array(data)[:,0]
y_1=np.array(data)[:,6]
y_2=np.array(data)[:,12]

# 绘制折线图
plt.plot(x, y_1,color=(242/255, 204/255, 142/255),label=r"$\mathbf{s}_1$")
plt.plot(x, y_2,color=(223/255, 122/255, 94/255),label=r"$\mathbf{s}_0$")
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(frameon=True, loc='upper right', )

# plt.title('')
plt.show()
