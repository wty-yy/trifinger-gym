import socket  
import json
import numpy as np
import torch
ADDRESS = ('192.168.1.143', 123)
# 如果开多个客户端，这个client_type设置不同的值，比如客户端1为linxinfa，客户端2为linxinfa2
client_type ='sim'
class Robot:
    def __init__(self) -> None:
        self.dof_pos=torch.zeros(9,dtype=torch.float32)
        self.cube_state=torch.zeros(7,dtype=torch.float32)
        self.target_cube_state=torch.zeros(7,dtype=torch.float32)
        
def tcp_init():
    client=socket.socket()
    client.connect(ADDRESS)
    print(client.recv(1024).decode(encoding='utf8'))
    send_data(client, 'CONNECT','0')
    return client
def send_data(client, cmd, kv):
    global client_type
    jd = {}
    jd['COMMAND'] = cmd
    jd['client_type'] = client_type
    jd['data'] = kv
    
    jsonstr = json.dumps(jd)
    #print('send: ' + jsonstr)
    client.sendall(jsonstr.encode('utf8'))


def input_client_type():
    return input("注册客户端，请输入名字 :")


trifinger=Robot()

def get_message(client):
    send_data(client,'SEND_DATA',0)
    msg=client.recv(1024).decode(encoding='utf8')
    
    jd=json.loads(msg)
    if jd['client_type']=='trifinger':
        data=jd['data']
        parts=data.replace('[','',1).replace(']','',1).replace('\n','').split(' ')
        #print("parts:",parts)
        data=[]
        for part in parts:
            if part !='':
                data.append(float(part))            
        print(data)
        if len(data)==23:
            trifinger.dof_pos=torch.tensor(data[0:9],dtype=torch.float32)
            trifinger.cube_state=torch.tensor(data[9:16],dtype=torch.float32)
            trifinger.target_cube_state=torch.tensor(data[16:23],dtype=torch.float32)

if '__main__' == __name__:
    client=tcp_init()
    import time
    while 1:
        send_data(client,'SEND_DATA',0)
        msg=client.recv(1024).decode(encoding='utf8')
        jd=json.loads(msg)
        if jd['client_type']=='trifinger':
            data=jd['data']
            parts=data.replace('[','',1).replace(']','',1).split(' ')
     
            data=[]
            for part in parts:
                data.append(float(part))            
            print(data)
            if len(data)==23:
                trifinger.dof_pos=torch.tensor(data[0:9],dtype=torch.float32)
                trifinger.cube_state=torch.tensor(data[9:16],dtype=torch.float32)
                trifinger.target_cube_state=torch.tensor(data[16:23],dtype=torch.float32)

        time.sleep(0.1)


