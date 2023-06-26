

import pandas as pd
import matplotlib.pyplot as plt
import  numpy as np
import  scienceplots
plt.style.use(['science','ieee'])

# 读取CSV文件并转换为Pandas DataFrame
data1 = pd.read_csv('agent1_diffusion_effect.csv')
x1=(np.array(data1)[:,0]-1)*10**-6
y1_1=np.array(data1)[:,6]
y1_2=np.array(data1)[:,12]

data2 = pd.read_csv('agent3_diffusion_effect.csv')
x3=(np.array(data2)[:,0]-1)*10**-6
y3_1=np.array(data2)[:,6]
y3_2=np.array(data2)[:,12]
# fig, axs = plt.subplots(1, 2, figsize=(7, 2))
# fig, axs = plt.subplots(1, 2)
# 绘制折线图
plt.plot(x1, y1_2,color=(223/255, 122/255, 94/255),label=r"with diffusion")
plt.plot(x1, y1_1,color=(130/255, 178/255, 154/255),label=r"without diffusion")
plt.xlabel(r'Iterations$(10^6)$')
plt.ylabel('Episode Return')
plt.legend(frameon=True, loc='right', )
# plt.set_title('1 end-device')
plt .savefig('agent1_diffusion_effect.png', dpi=300, bbox_inches='tight')
plt.clf()


plt.plot(x3, y3_2,color=(223/255, 122/255, 94/255),label=r"with diffusion")
plt.plot(x3, y3_1,color=(130/255, 178/255, 154/255),label=r"without diffusion")
plt.xlabel(r'Iterations$(10^6)$')
plt.ylabel('Episode Return')
plt.legend(frameon=True, loc='right', )
# plt.set_title('3 end-devices')
# plt.title('')
plt.savefig('agent3_diffusion_effect.png', dpi=300, bbox_inches='tight')
plt.show()
