# import pandas as pd
# import matplotlib.pyplot as plt
# import  numpy as np
# import  scienceplots
# plt.style.use(['science','ieee'])
#
# # 读取CSV文件并转换为Pandas DataFrame
# data = pd.read_csv('agent1_diffusion_effect.csv')
# x=np.array(data)[:,0]*10**-6
# y_1=np.array(data)[:,6]
# y_2=np.array(data)[:,12]
#
# # 绘制折线图
# plt.plot(x, y_1,color=(130/255, 178/255, 154/255),label=r"without diffusion")
# plt.plot(x, y_2,color=(223/255, 122/255, 94/255),label=r"with diffusion")
# plt.xlabel(r'Iterations$(10^6)$')
# plt.ylabel('Episode Return')
# plt.legend(frameon=True, loc='right', )
#
# # plt.title('')
# plt.savefig('agent1_diffusion_effect.pdf', dpi=300, bbox_inches='tight')
# plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
# import  numpy as np
# import  scienceplots
# plt.style.use(['science','ieee'])
#
# # 读取CSV文件并转换为Pandas DataFrame
# data = pd.read_csv('agent3_diffusion_effect.csv')
# x=np.array(data)[:,0]*10**-6
# y_1=np.array(data)[:,6]
# y_2=np.array(data)[:,12]
#
# # 绘制折线图
# plt.plot(x, y_1,color=(130/255, 178/255, 154/255),label=r"without diffusion")
# plt.plot(x, y_2,color=(223/255, 122/255, 94/255),label=r"with diffusion")
# plt.xlabel(r'Iterations$(10^6)$')
# plt.ylabel('Episode Return')
# plt.legend(frameon=True, loc='right', )
#
# # plt.title('')
# plt.savefig('agent1_diffusion_effect.pdf', dpi=300, bbox_inches='tight')
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import  numpy as np
import  scienceplots
plt.style.use(['science','ieee'])

# 读取CSV文件并转换为Pandas DataFrame
data1 = pd.read_csv('agent1_diffusion_effect.csv')
data3 = pd.read_csv('agent3_diffusion_effect.csv')
x1=np.array(data1)[:,0]*10**-6
y1_1=np.array(data1)[:,6]
y1_2=np.array(data1)[:,12]

x3=np.array(data3)[:,0]*10**-6
y3_1=np.array(data3)[:,6]
y3_2=np.array(data3)[:,12]
fig, axs = plt.subplots(1, 2, figsize=(7, 2))
# fig, axs = plt.subplots(1, 2)
# 绘制折线图
axs[0].plot(x1, y1_1,color=(130/255, 178/255, 154/255),label=r"without diffusion")
axs[0].plot(x1, y1_2,color=(223/255, 122/255, 94/255),label=r"with diffusion")
axs[0].set_xlabel(r'Iterations$(10^6)$')
axs[0].set_ylabel('Episode Return')
axs[0].legend(frameon=True, loc='right', )
axs[0].set_title('1 end-device')

axs[1].plot(x3, y3_1,color=(130/255, 178/255, 154/255),label=r"without diffusion")
axs[1].plot(x3, y3_2,color=(223/255, 122/255, 94/255),label=r"with diffusion")
axs[1].set_xlabel(r'Iterations$(10^6)$')
axs[1].set_ylabel('Episode Return')
axs[1].legend(frameon=True, loc='right', )
axs[1].set_title('3 end-devices')
# plt.title('')
plt.savefig('agent13_diffusion_effect.pdf', dpi=300, bbox_inches='tight')
plt.show()