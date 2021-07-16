#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


import torch
import torch.nn as nn

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"# DEVICE {i}: {torch.cuda.get_device_name(i)}")
        print("- Memory Usage:")
        print(f"  Allocated: {round(torch.cuda.memory_allocated(i)/1024**3,1)} GB")
        print(f"  Cached:    {round(torch.cuda.memory_cached(i)/1024**3,1)} GB\n")
        
else:
    print("# GPU is not available")


# In[3]:


# GPU 할당 변경하기
GPU_NUM = 1 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU

print ('# Current cuda device: ', torch.cuda.current_device()) # check


# In[4]:


if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print(f"using cuda: {GPU_NUM}, {torch.cuda.get_device_name(GPU_NUM)}")


# In[5]:


from torch.utils.data import Dataset

class MnistDataset(Dataset):
    def __init__(self, csv_file):
        self.data_df = pd.read_csv(csv_file, header=None)
        pass
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, index):
        label = self.data_df.iloc[index, 0]
        target = torch.zeros((10))
        target[label] = 1.0
        
        image_values = torch.Tensor(self.data_df.iloc[index, 1:].values)/255.0
        
        return label, image_values, target
    
    def plot_image(self, index):
        img = self.data_df.iloc[index, 1:].values.reshape(28, 28)
        plt.title(f"label = {self.data_df.iloc[index, 0]}")
        plt.imshow(img, interpolation='none', cmap='Blues')
        


# In[6]:


mnist_dataset = MnistDataset("../myo_gan/mnist_train.csv")
mnist_dataset.plot_image(7)


# In[7]:


def generate_random_image(size):
    random_data = torch.rand(size)
    return random_data

def generate_random_seed(size):
    random_data = torch.randn(size)
    return random_data


# In[8]:


generate_random_image(10)


# In[9]:


generate_random_seed(10)


# In[10]:


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )
        self.model = self.model.cuda()
        
        self.loss_function = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        
        self.counter = 0
        self.progress = []
        
    def forward(self, inputs):
        return self.model(inputs)
    
    def train(self, inputs, targets):
        outputs = self.forward(inputs)
        
        loss = self.loss_function(outputs, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())
        if self.counter % 10000 == 0:
            print(f"counter= {self.counter}")
            
    def plot_progress(self):
        df = pd.DataFrame(self.progress, columns=["loss"])
        df.plot(ylim=(0), figsize=(16, 8), alpha=0.1, marker=".", grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))


# In[11]:


D = Discriminator()

for label, image_data_tensor, target_tensor in mnist_dataset:
    # real data
    D.train(image_data_tensor, torch.Tensor([1.0]))
    
    # fake data
    D.train(generate_random_image(784), torch.Tensor([0.0]))
    


# In[12]:


# D.plot_progress()


# In[13]:


import random

for i in range(4):
    image_data_tensor = mnist_dataset[random.randint(0, 60000)][1]
    print(D.forward(image_data_tensor).item())


# In[14]:


for i in range(4):
    print(D.forward(generate_random_image(784)).item())


# In[15]:


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(100, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200, 784),
            nn.Sigmoid()
        )
        self.model = self.model.cuda()
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        
        self.counter = 0
        self.progress = []
        
    def forward(self, inputs):
        return self.model(inputs)
    
    def train(self, D, inputs, targets):
        g_output = self.forward(inputs)
        
        d_output = D.forward(g_output)
        
        loss = D.loss_function(d_output, targets)
        
        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def plot_progress(self):
        df = pd.DataFrame(self.progress, columns=["loss"])
        df.plot(ylim=(0), figsize=(16, 8), alpha=0.1, marker=".", grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))
        


# In[16]:


G = Generator()

output = G.forward(generate_random_seed(100))
img = output.cpu().detach().numpy().reshape(28, 28)
plt.imshow(img, cmap="Blues")


# In[17]:


#get_ipython().run_cell_magic('time', '', '\nD = Discriminator()\nG = Generator()\n\nepochs = 4\n\nfor epoch in range(epochs):\n    print(f"epoch= {epoch+1}")\n    \n    for label, image_data_tensor, target_tensor in mnist_dataset:\n        D.train(image_data_tensor, torch.Tensor([1.0]))\n        D.train(G.forward(generate_random_seed(100)).detach(), torch.Tensor([0.0]))\n        G.train(D, generate_random_seed(100), torch.Tensor([1.0]))')


# In[18]:


#D.plot_progress()


# In[19]:


#G.plot_progress()


# In[20]:


#fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(16, 8))
#for ax in axes.ravel():
#    output = G.forward(generate_random_seed(100))
#    img = output.cpu().detach().numpy().reshape(28, 28)
#    ax.imshow(img, cmap="Blues")


# In[21]:


### conditional GAN


# In[22]:


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(784+10, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )
        self.model = self.model.cuda()
        
        self.loss_function = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        
        self.counter = 0
        self.progress = []
        
    def forward(self, image_tensor, label_tensor):
        inputs = torch.cat((image_tensor, label_tensor))
        return self.model(inputs)
    
    def train(self, inputs, label_tensor, targets):
        outputs = self.forward(inputs, label_tensor)
        
        loss = self.loss_function(outputs, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())
        if self.counter % 10000 == 0:
            print(f"counter= {self.counter}")
            
    def plot_progress(self):
        df = pd.DataFrame(self.progress, columns=["loss"])
        df.plot(ylim=(0), figsize=(16, 8), alpha=0.1, marker=".", grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))


# In[23]:


def generate_random_one_hot(size):
    label_tensor = torch.zeros((size))
    random_idx = random.randint(0, size-1)
    label_tensor[random_idx] = 1.0
    return label_tensor


# In[24]:


D = Discriminator()

for label, image_data_tensor, label_tensor in mnist_dataset:
    D.train(image_data_tensor, label_tensor, torch.Tensor([1.0]))
    D.train(generate_random_image(784), generate_random_one_hot(10), torch.Tensor([0.0]))


# In[25]:


#D.plot_progress()


# In[26]:


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(100+10, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200, 784),
            nn.Sigmoid()
        )
        self.model = self.model.cuda()
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        
        self.counter = 0
        self.progress = []
        
    def forward(self, seed_tensor, label):
        inputs = torch.cat((seed_tensor, label))
        return self.model(inputs)
    
    def train(self, D, inputs, label_tensor, targets):
        g_output = self.forward(inputs, label_tensor)
        
        d_output = D.forward(g_output, label_tensor)
        
        loss = D.loss_function(d_output, targets)
        
        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def plot_progress(self):
        df = pd.DataFrame(self.progress, columns=["loss"])
        df.plot(ylim=(0), figsize=(16, 8), alpha=0.1, marker=".", grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))
        
    def plot_images(self, label):
        label_tensor = torch.zeros((10))
        label_tensor[label] = 1.0

        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        for ax in axes.ravel():
            ax.imshow(G.forward(generate_random_seed(100), label_tensor).detach().cpu().numpy().reshape(28, 28), cmap="Blues")


# In[27]:


D = Discriminator()
G = Generator()

epochs = 12
for epoch in range(epochs):
    for label, image_data_tensor, label_tensor in mnist_dataset:
        D.train(image_data_tensor, label_tensor, torch.Tensor([1.0]))

        random_label = generate_random_one_hot(10)
        D.train(G.forward(generate_random_seed(100), random_label).detach(), random_label, torch.Tensor([0.0]))

        random_label = generate_random_one_hot(10)
        G.train(D, generate_random_seed(100), random_label, torch.Tensor([1.0]))
    


# In[28]:


#D.plot_progress()


# In[ ]:


#G.plot_progress()


# In[ ]:


#G.plot_images(9)


# In[ ]:





# In[ ]:


# generator class

class Generator(nn.Module):
    
    def __init__(self):
        # initialise parent pytorch class
        super().__init__()
        
        # define neural network layers
        self.model = nn.Sequential(
            nn.Linear(100+10, 200),
            nn.LeakyReLU(0.02),

            nn.LayerNorm(200),

            nn.Linear(200, 784),
            nn.Sigmoid()
        )
        self.model = self.model.cuda()
        
        # create optimiser, simple stochastic gradient descent
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

        # counter and accumulator for progress
        self.counter = 0;
        self.progress = []
        
        pass
    
    
    def forward(self, seed_tensor, label_tensor):        
        # combine seed and label
        inputs = torch.cat((seed_tensor, label_tensor))
        return self.model(inputs)


    def train(self, D, inputs, label_tensor, targets):
        # calculate the output of the network
        g_output = self.forward(inputs, label_tensor)
        
        # pass onto Discriminator
        d_output = D.forward(g_output, label_tensor)
        
        # calculate error
        loss = D.loss_function(d_output, targets)

        # increase counter and accumulate error every 10
        self.counter += 1;
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass

        # zero gradients, perform a backward pass, update weights
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        pass
    
    def plot_images(self, label):
        label_tensor = torch.zeros((10))
        label_tensor[label] = 1.0
        # plot a 3 column, 2 row array of sample images
        f, axarr = plt.subplots(2,3, figsize=(16,8))
        for i in range(2):
            for j in range(3):
                axarr[i,j].imshow(G.forward(generate_random_seed(100), label_tensor).detach().cpu().numpy().reshape(28,28), interpolation='none', cmap='Blues')
                pass
            pass
        pass
    
    def plot_progress(self):
        df = pd.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))
        pass
    
    pass


# In[ ]:


# create Discriminator and Generator

D = Discriminator()
G = Generator()


# In[ ]:


#get_ipython().run_cell_magic('time', '', '\n# train Discriminator and Generator\n\nepochs = 12\n\nfor epoch in range(epochs):\n    print ("epoch = ", epoch + 1)\n\n    # train Discriminator and Generator\n\n    for label, image_data_tensor, label_tensor in mnist_dataset:\n        # train discriminator on true\n        D.train(image_data_tensor, label_tensor, torch.Tensor([1.0]))\n\n        # random 1-hot label for generator\n        random_label = generate_random_one_hot(10)\n\n        # train discriminator on false\n        # use detach() so gradients in G are not calculated\n        D.train(G.forward(generate_random_seed(100), random_label).detach(), random_label, torch.Tensor([0.0]))\n\n        # different random 1-hot label for generator\n        random_label = generate_random_one_hot(10)\n\n        # train generator\n        G.train(D, generate_random_seed(100), random_label, torch.Tensor([1.0]))\n\n        pass\n\n    pass')


# In[ ]:


#G.plot_images(9)


# In[ ]:




