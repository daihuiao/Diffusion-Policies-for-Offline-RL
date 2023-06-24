import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use(['science', 'ieee'])

# 读取CSV文件并转换为Pandas DataFrame
data1 = pd.read_csv('FDQLworks.csv')
x1 = (np.array(data1)[:, 0]-1) * 10 ** -6
y1_1 = np.array(data1)[:, 1]
y1_2 = np.array(data1)[:, 4]
y1_3 = np.array(data1)[:, 7]

# fig, axs = plt.subplots(1, 2)
# 绘制折线图
plt.plot(x1, y1_3, color=(223 / 255, 122 / 255, 94 / 255), label=r"1 end-device")
plt.plot(x1, y1_2, color=(130 / 255, 178 / 255, 154 / 255), label=r"3 end-devices")
plt.plot(x1, y1_1, color=(60/255, 64/255, 91/255), label=r"20 end-devices")
plt.xlabel(r'Iterations$(10^6)$')
plt.ylabel('Episode Return')
plt.legend(frameon=True, loc='lower right', )

# plt.title('')
plt.savefig('FDQL.pdf', dpi=300, bbox_inches='tight')
plt.show()
