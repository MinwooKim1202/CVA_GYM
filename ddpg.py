#!/usr/bin/env python3
import cva_gym
import random
import time
import collections
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
lr_mu        = 0.0005
lr_q         = 0.0003
gamma        = 0.99
batch_size   = 32
buffer_limit = 50000
tau          = 0.005 # for target network soft update

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            a = a.tolist()
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        
        
        #return_a =  torch.tensor(a_lst, dtype=torch.float)
        return_a = np.reshape(a_lst, (batch_size, 2)) 
        #print("return_a :{}" .format(return_a.shape))

        return_a =  torch.tensor(return_a, dtype=torch.float)
        

        return torch.tensor(s_lst, dtype=torch.float), return_a, \
                torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                torch.tensor(done_mask_lst, dtype=torch.float)
    
    def size(self):
        return len(self.buffer)

class MuNet(nn.Module):
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(360, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.fc_mu = nn.Linear(64, 2)

    def forward(self, x):
        #x = x.unsqueeze(0)
        x = self.fc1(x)
        #x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        #x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc_mu(x)
        mu_lst = [] 
        #print("x:{} " .format(x))
        if len(x) > 2:
            #print("batch cal!!!")
                       
            for i in range(batch_size):
                #print(x[i])
                x1, x2 = x[i]
                #print("x1:{}" .format(x1))
                #print("x2:{}" .format(x2))
                linear = F.relu(x1)
                #linear = torch.tanh(x1)*2
                
                twist = torch.tanh(x2)*3

                #print(linear)
                #print(twist)
                mu = torch.FloatTensor([linear, twist])
                mu = mu.tolist()
                mu_lst.append(mu)
                #print("mu list add!!")
            
            return_mu =  torch.tensor(mu_lst, dtype=torch.float)
            return return_mu

        else:
            linear = F.relu(x[0])
            #linear = torch.tanh(x[0])*2
            twist = torch.tanh(x[1])*3
            mu = torch.FloatTensor([linear, twist])
            return mu
        #print(mu_lst)
        #print(len(mu_lst))

        #print("linear:{}".format(linear))
        #print("twist:{}".format(twist))
        #mu = torch.FloatTensor([linear, twist])
        #print(mu)
        #mu = torch.tanh(self.fc_mu(x))*2 # Multipled by 2 because the action space of the Pendulum-v0 is [-2,2]
        #return mu

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(360, 64)
        self.fc_a = nn.Linear(2,64)
        self.fc_q = nn.Linear(128, 128)
        self.fc_out = nn.Linear(128,1)
        self.bn_s = nn.BatchNorm1d(64)
        self.bn_a = nn.BatchNorm1d(64)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x, a):
        #print(x.shape)
        #print(a.shape)
        x = self.fc_s(x)
        #x = self.bn_s(x)
        h1 = F.relu(x)
        h1 = self.dropout(h1)

        a = self.fc_a(a)
        #a = self.bn_a(a)
        h2 = F.relu(a)
        h2 = self.dropout(h2)
        #print(h1.shape)
        #print(h2.shape)
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.dropout(q)
        q = self.fc_out(q)
        return q

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
      
def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    s,a,r,s_prime,done_mask  = memory.sample(batch_size)
           
    target = r + gamma * q_target(s_prime, mu_target(s_prime)) * done_mask

    q_loss = F.smooth_l1_loss(q(s,a), target.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()
    #print("q_loss : {}" .format(q_loss))
    
    mu_loss = -q(s,mu(s)).mean() # That's all for the policy loss.
    #print("mu_loss : {}" .format(mu_loss))
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()
    
def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
    
def main():
    env = cva_gym.make('simple_circuit')
    memory = ReplayBuffer()

    q, q_target = QNet(), QNet()
    q_target.load_state_dict(q.state_dict())
    mu, mu_target = MuNet(), MuNet()
    mu_target.load_state_dict(mu.state_dict())

    score = 0.0
    print_interval = 10

    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    q_optimizer  = optim.Adam(q.parameters(), lr=lr_q)
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

    for n_epi in range(10000):
        s = env.reset()
        done = False
        
        while not done:
            a = mu(torch.from_numpy(s).float())
            
            linear = a[0] + (ou_noise()[0] / 5)
            twist =  a[1] + (ou_noise()[0] / 2) 

            s_prime, r, done = env.step([linear, twist])
            time.sleep(0.02)
            memory.put((s,a,r ,s_prime,done))
            score +=r
            s = s_prime

                
        if memory.size()>2000:
            for i in range(10):
                train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
                soft_update(mu, mu_target)
                soft_update(q,  q_target)
        
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()
