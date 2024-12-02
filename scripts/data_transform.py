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
from scipy.ndimage.interpolation import shift

# root_path='/home/wq/Documents'
# file_path=['dataset_v3_2.csv','dataset_v4.csv','dataset_v5.csv','dataset_v6.csv']
# i=1
# for file in file_path:
#     datapath=root_path+'/'+file
#     print(datapath)
#     csv_file=pd.read_csv(datapath)
#     print(csv_file.shape)
#     length=csv_file.shape[0]//750 *750 +1
#     print("length:",length)
#     dic=dict()
#     dic['observations']=np.array(csv_file.iloc[:length,0:41].values)
#     dic['actions']=np.array(csv_file.iloc[:length,41:50].values)
#     dic['rewards']=np.array(csv_file.iloc[:length,50:51].values)
#     dic['terminals']=np.array(csv_file.iloc[:length,51:52].values)
#     dic['next_observations']=np.array(csv_file.iloc[:length,52:93].values)



#     index=np.argwhere(dic['terminals']==1)
#     index=index[:,0]
#     print(index)

#     dic['rewards']=np.roll(dic['rewards'],1,0)  # the reward and terminal belong to the nex_obs
#     dic['terminals']=np.roll(dic['terminals'],1,0)
#     dic['rewards'][0]=0
#     dic['terminals'][0]=0
#     index=np.argwhere(dic['terminals']==1)
#     index=index[:,0]
#     print(index)

#     data=dict()
#     data['observations']=dic['observations'][:1,:]
#     data['actions']=dic['actions'][:1,:]
#     data['rewards']=dic['rewards'][:1,:]
#     data['terminals']=dic['terminals'][:1,:]
#     data['next_observations']=dic['next_observations'][:1,:]
#     idx_head=1
#     for idx_tail in tqdm(index,desc="dealing"):
#         #print(idx_tail)
#         if (idx_tail)%750==0:  #  reset after 750 actions
#             idx_head=idx_tail+1
#         else:
#             data['observations']=np.concatenate((data['observations'],dic['observations'][idx_head:(idx_tail+1),:]))
#             data['actions']=np.concatenate((data['actions'],dic['actions'][idx_head:(idx_tail+1),:]))
#             data['rewards']=np.concatenate((data['rewards'],dic['rewards'][idx_head:(idx_tail+1),:]))
#             data['terminals']=np.concatenate((data['terminals'],dic['terminals'][idx_head:(idx_tail+1),:]))
#             data['next_observations']=np.concatenate((data['next_observations'],dic['next_observations'][idx_head:(idx_tail+1),:]))
#             idx_head=idx_tail+1
#     print("data_shape:",data['observations'].shape)
#     idx=np.argwhere(data['terminals']==1)[:,0]
#     np.savetxt("test2.txt",idx)
#     print(f"rew_max:{np.max(data['rewards'])},min:{np.min(data['rewards'])}")
#     print(f"obs_max:{np.max(data['observations'])},min:{np.min(data['observations'])}")
#     print(f"act_max:{np.max(data['actions'])},min:{np.min(data['actions'])}")
#     np.save(f'mydata_v4_{i}',data)
#     i+=1

data_1=np.load("/home/wq/Documents/mydata_v4_1.npy",allow_pickle=True).item()
data_2=np.load("/home/wq/Documents/mydata_v4_2.npy",allow_pickle=True).item()
data_3=np.load("/home/wq/Documents/mydata_v4_3.npy",allow_pickle=True).item()
data_4=np.load("/home/wq/Documents/mydata_v4_4.npy",allow_pickle=True).item()
data=dict()
print(data_1["observations"].shape)
print(data_2["observations"].shape)
print(data_3["observations"].shape)
print(data_4["observations"].shape)
for i in ['observations','actions','rewards','terminals','next_observations']:
    print(i)
    data[i]=np.concatenate((data_1[i],data_2[i]))
for i in ['observations','actions','rewards','terminals','next_observations']:
    print(i)
    data[i]=np.concatenate((data[i],data_3[i]))

for i in ['observations','actions','rewards','terminals','next_observations']:
    print(i)
    data[i]=np.concatenate((data[i],data_4[i]))
print(data["observations"].shape)
np.save("data_v4_1367861",data)