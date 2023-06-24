

import pandas as pd
import matplotlib.pyplot as plt
import  numpy as np
import  scienceplots
plt.style.use(['science','ieee'])

# 读取CSV文件并转换为Pandas DataFrame
data1 = pd.read_csv('agent13_res_effect.csv')
x1=np.array(data1)[:,0]*10**-6-1
y1_1=np.array(data1)[:,10]
y1_2=np.array(data1)[:,4]

x3=np.array(data1)[:,0]*10**-6-1
y3_1=np.array(data1)[:,7]
y3_2=np.array(data1)[:,1]
fig, axs = plt.subplots(1, 2, figsize=(7, 2))
# fig, axs = plt.subplots(1, 2)
# 绘制折线图
axs[0].plot(x1, y1_2,color=(223/255, 122/255, 94/255),label=r"with perturbation network")
axs[0].plot(x1, y1_1,color=(130/255, 178/255, 154/255),label=r"without perturbation network")
axs[0].set_xlabel(r'Iterations$(10^6)$')
axs[0].set_ylabel('Episode Return')
axs[0].legend(frameon=True, loc='lower right', )
axs[0].set_title('1 end-device')

axs[1].plot(x3, y3_2,color=(223/255, 122/255, 94/255),label=r"with perturbation network")
axs[1].plot(x3, y3_1,color=(130/255, 178/255, 154/255),label=r"without perturbation network")
axs[1].set_xlabel(r'Iterations$(10^6)$')
axs[1].set_ylabel('Episode Return')
axs[1].legend(frameon=True, loc='lower right', )
axs[1].set_title('3 end-devices')
# plt.title('')
plt.savefig('agent13_res_effect.pdf', dpi=300, bbox_inches='tight')
plt.show()