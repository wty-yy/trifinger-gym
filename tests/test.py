import torch
import csv
import numpy as np
import random
import pandas as pd
import time
import h5py
import json
import matplotlib.pyplot as plt
import seaborn as sns
print(torch.cuda.is_available())
from tqdm import tqdm
# 返回True 接着用下列代码进一步测试

a=torch.tensor([[1.2,2,3,4],[5,6,7,8]])
b=torch.tensor([[2,3,3,3],[4,5,6,7]])
c=torch.abs(a-b)<1
print(c)
print(a*~c)
print(torch.nonzero(c))
if torch.all(c):
    print("false")
else:
    print(233)

  
csv_file=pd.read_csv('/home/wq/Documents/dataset_v4.csv')
print(csv_file.shape)
dic=dict()
dic['observations']=np.array(csv_file.iloc[:663750,0:41].values)
dic['actions']=np.array(csv_file.iloc[:663750,41:50].values)
dic['rewards']=np.array(csv_file.iloc[:663750,50:51].values)
dic['terminals']=np.array(csv_file.iloc[:663750,51:52].values)
dic['next_observations']=np.array(csv_file.iloc[:663750,52:93].values)

index=np.argwhere(dic['terminals']==1)
index=index[:,0]
print(index)
# data=dict()
# data['observations']=np.array(csv_file.iloc[:1,0:41].values)
# data['actions']=np.array(csv_file.iloc[:1,41:50].values)
# data['rewards']=np.array(csv_file.iloc[:1,50:51].values)
# data['terminals']=np.array(csv_file.iloc[:1,51:52].values)
# data['next_observations']=np.array(csv_file.iloc[:1,52:93].values)
# idx_head=1
# for idx_tail in tqdm(index):
#     #print(idx_tail)
#     if (idx_tail+1)%750==0:
#         idx_head=idx_tail+1
#     else:
#         data['observations']=np.concatenate((data['observations'],np.array(csv_file.iloc[idx_head:(idx_tail+1),0:41].values)))
#         data['actions']=np.concatenate((data['actions'],np.array(csv_file.iloc[idx_head:(idx_tail+1),41:50].values)))
#         data['rewards']=np.concatenate((data['rewards'],np.array(csv_file.iloc[idx_head:(idx_tail+1),50:51].values)))
#         data['terminals']=np.concatenate((data['terminals'],np.array(csv_file.iloc[idx_head:(idx_tail+1),51:52].values)))
#         data['next_observations']=np.concatenate((data['next_observations'],np.array(csv_file.iloc[idx_head:(idx_tail+1),52:93].values)))
#         idx_head=idx_tail+1
# print(data['observations'].shape)
# idx=np.argwhere(data['terminals']==1)[:,0]
# np.savetxt("test2.txt",idx)
# print(f"rew_max:{np.max(data['rewards'])},min:{np.min(data['rewards'])}")
# print(f"obs_max:{np.max(data['observations'])},min:{np.min(data['observations'])}")
# print(f"act_max:{np.max(data['actions'])},min:{np.min(data['actions'])}")
#np.save('mydata_v3_93',dic)

#print(index)
#np.savetxt("test.txt",index)


# csv_file=pd.read_csv('/home/wq/Documents/dataset_v2.csv')

# print(csv_file.shape)
# dic=dict()
# dic['observations']=np.array(csv_file.iloc[:,0:41].values)
# dic['actions']=np.array(csv_file.iloc[:,41:50].values)
# dic['terminals']=np.array(csv_file.iloc[:,51:52].values)
# dic['rewards']=np.array(csv_file.iloc[:,50:51].values)
# print(f"obs_max:{np.max(dic['observations'])},min:{np.min(dic['observations'])}")
# print(f"act_max:{np.max(dic['actions'])},min:{np.min(dic['actions'])}")
# print(f"rew_max:{np.max(dic['rewards'])},min:{np.min(dic['rewards'])}")
# print(np.argwhere(dic['observations']<-1.5))
# sns.displot(x=dic['observations'].flatten(),kind='kde')
# plt.show()

# print([str(x) for x in range(93)])
# print(np.argwhere(dic['terminals']>0.51))

# data=np.load('madata_460000_93.npy',allow_pickle=True).item()
# print(data['rewards'].shape)

# x=range(0,data['rewards'][data['rewards']<4000].shape[0])
# print(len(x))
# fig, ax = plt.subplots()
# # 在生成的坐标系下画折线图
# ax.plot(x, data['rewards'][data['rewards']<4000], 'red', linewidth=1)

# print(np.argwhere(data['terminals']>0.51))
# x=range(0,data['terminals'].shape[0])

# fig, ax = plt.subplots()
# # 在生成的坐标系下画折线图
# ax.plot(x, data['terminals'], 'red', linewidth=1)
# # 显示图形
# plt.show()

# start=time.time()
# csv_file=pd.read_csv('/home/wq/Documents/dataset_v2.csv')
# print(time.time()-start)
# print(csv_file.shape)
# dic=dict()
# dic['observations']=np.array(csv_file.iloc[:540000,0:41].values)
# dic['actions']=np.array(csv_file.iloc[:540000,41:50].values)
# dic['rewards']=np.array(csv_file.iloc[:540000,50:51].values)
# dic['terminals']=np.array(csv_file.iloc[:540000,51:52].values)
# dic['next_observations']=np.array(csv_file.iloc[:540000,52:93].values)
# print(dic['rewards'].shape)
# for i in range(len(dic['rewards'])):
#     if dic['rewards'][i,0]>4000:
#         dic['rewards'][i,0]-=4999
# print(f"rew_max:{np.max(dic['rewards'])},min:{np.min(dic['rewards'])}")
# np.save('mydata_540000_93',dic)




