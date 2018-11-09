"""
    Cheryl Liao
    CSC578-701
    Homework #2: Backprop Hyper-parameters
  """""""""""""""
# coding: utf-8

# In[1]:


import NN578_network2 as network2
import numpy as np
# Load the iris train-test (separate) data files
def my_load_csv(fname, no_trainfeatures, no_testfeatures):
    ret = np.genfromtxt(fname, delimiter=',')
    data = np.array([(entry[:no_trainfeatures],entry[no_trainfeatures:]) for entry in ret])
    temp_inputs = [np.reshape(x, (no_trainfeatures, 1)) for x in data[:,0]] 
    temp_results = [np.reshape(y, (no_testfeatures, 1)) for y in data[:,1]] 
    dataset = list(zip(temp_inputs, temp_results))
    return dataset
iris_train = my_load_csv('../data/iris-train-1.csv', 4, 3)
iris_test = my_load_csv('../data/iris-test-1.csv', 4, 3)


# In[2]:


# Test with one-data Iris data
inst1 = (np.array([5.7, 3, 4.2, 1.2]), np.array([0., 1., 0.]))
x1 = np.reshape(inst1[0], (4, 1))
y1 = np.reshape(inst1[1], (3, 1))
sample1 = [(x1, y1)]
inst2 = (np.array([4.8, 3.4, 1.6, 0.2]), np.array([1., 0., 0.]))
x2 = np.reshape(inst2[0], (4, 1))
y2 = np.reshape(inst2[1], (3, 1))
sample2 = [(x2, y2)]
net4 = network2.load_network("iris-423.dat")
net4.set_parameters(cost=network2.QuadraticCost)
net4.SGD(sample1, 2, 1, 1.0, evaluation_data=sample2, monitor_evaluation_cost=True,
monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)


# In[3]:


net2 = network2.load_network("iris-423.dat")
# Set hyper-parameter values individually after the network
net2.set_parameters(cost=network2.QuadraticCost)
net2.SGD(iris_train, 10, 1, 1.0, evaluation_data=iris_test, monitor_evaluation_cost=True,
monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)


# In[4]:


net3 = network2.load_network("iris-423.dat")
# Set hyper-parameter values individually after the network
net3.set_parameters(cost=network2.CrossEntropyCost)
net3.SGD(iris_train, 10, 1, 1.0, evaluation_data=iris_test, monitor_evaluation_cost=True,
monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)


# In[5]:


net1 = network2.load_network("iris-423.dat")
# Set hyper-parameter values individually after the network
net1.set_parameters(cost=network2.LogLikelihood)
net1.SGD(iris_train, 10, 1, 1.0, evaluation_data=iris_test, monitor_evaluation_cost=True,
monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)


# In[6]:


net3 = network2.load_network("iris-423.dat")
# Set hyper-parameter values individually after the network
net3.set_parameters(cost=network2.LogLikelihood, act_output=network2.Softmax)
net3.SGD(iris_train, 10, 1, 1.0, evaluation_data=iris_test, monitor_evaluation_cost=True,
monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)


# In[8]:


net1 = network2.load_network("iris-423.dat")
# Set hyper-parameter values individually after the network
net1.set_parameters(cost=network2.LogLikelihood, act_output=network2.Softmax, act_hidden=network2.ReLU)
net1.SGD(iris_train, 10, 1, 1.0, evaluation_data=iris_test, monitor_evaluation_cost=True,
monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)


# In[9]:


net6 = network2.load_network("iris-423.dat")
# Set hyper-parameter values individually after the network
net6.set_parameters(cost=network2.CrossEntropyCost, act_output=network2.Softmax,
act_hidden=network2.ReLU)
net6.SGD(iris_train, 10, 1, 1.0, evaluation_data=iris_test, monitor_evaluation_cost=True,
monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)


# In[10]:


net3 = network2.load_network("iris-423.dat")
# Set hyper-parameter values individually after the network
net3.set_parameters(cost=network2.CrossEntropyCost, act_output=network2.Sigmoid,
act_hidden=network2.Tanh)
net3.SGD(iris_train, 10, 1, 1.0, evaluation_data=iris_test, monitor_evaluation_cost=True,
monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)


# In[12]:


net2 = network2.load_network("iris-423.dat")
# Set hyper-parameter values individually after the network
net2.set_parameters(cost=network2.CrossEntropyCost, act_output=network2.Tanh, act_hidden=network2.Tanh)
net2.SGD(iris_train, 10, 1, 1.0, evaluation_data=iris_test, monitor_evaluation_cost=True,
monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)


# In[13]:


net2 = network2.load_network("iris-423.dat")
# Set hyper-parameter values individually after the network
net2.set_parameters(cost=network2.QuadraticCost, regularization=network2.L2)
net2.SGD(iris_train, 10, 1, 1.0, lmbda = 3.0, evaluation_data=iris_test, monitor_evaluation_cost=True, monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)


# In[14]:


net1 = network2.load_network("iris-423.dat")
# Set hyper-parameter values individually after the network
net1.set_parameters(cost=network2.QuadraticCost, regularization=network2.L1)
net1.SGD(iris_train, 10, 1, 1.0, lmbda = 3.0, evaluation_data=iris_test, monitor_evaluation_cost=True, monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)


# In[15]:


net1 = network2.load_network("iris-423.dat")
# Set hyper-parameter values individually after the network
net1.set_parameters(cost=network2.LogLikelihood, act_hidden=network2.ReLU,
                    act_output=network2.Softmax, regularization=network2.L2)
net1.SGD(iris_train, 10, 1, 1.0, lmbda = 3.0, evaluation_data=iris_test, monitor_evaluation_cost=True, monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)


# In[16]:


net2 = network2.load_network("iris-4-20-7-3.dat")
# Set hyper-parameter values individually after the network
net2.set_parameters(cost=network2.QuadraticCost)
net2.SGD(iris_train, 10, 1, 1.0, evaluation_data=iris_test, monitor_evaluation_cost=True,
monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)


# In[17]:


net2 = network2.load_network("iris-4-20-7-3.dat")
# Set hyper-parameter values individually after the network
net2.set_parameters(cost=network2.QuadraticCost,dropoutpercent=0.3)
net2.SGD(iris_train, 10, 1, 1.0, evaluation_data=iris_test, monitor_evaluation_cost=True,
         monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)


# In[18]:


net2 = network2.load_network("iris-4-20-7-3.dat")
# Set hyper-parameter values individually after the network
net2.set_parameters(cost=network2.QuadraticCost,dropoutpercent=0.5)
net2.SGD(iris_train, 10, 1, 1.0, evaluation_data=iris_test, monitor_evaluation_cost=True,
         monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)
#[19]ReLU  Softmax  LogLikelihood    (default)    0.0    0.1:
net1 = network2.load_network("iris-4-20-7-3.dat")
# Set hyper-parameter values individually after the network
net1.set_parameters(cost=network2.LogLikelihood, act_hidden=network2.ReLU, act_output=network2.Softmax,dropoutpercent=0.1)
net1.SGD(iris_train, 10, 1, 1.0, evaluation_data=iris_test, monitor_evaluation_cost=True,
         monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)
[20]
net1 = network2.load_network("iris-4-20-7-3.dat")
# Set hyper-parameter values individually after the network
net1.set_parameters(cost=network2.LogLikelihood, act_hidden=network2.ReLU, act_output=network2.Softmax,dropoutpercent=0.5)
net1.SGD(iris_train, 10, 1, 1.0, evaluation_data=iris_test, monitor_evaluation_cost=True,
         monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)
[21]
net1 = network2.load_network("iris-4-20-7-3.dat")
# Set hyper-parameter values individually after the network
net1.set_parameters(cost=network2.LogLikelihood, act_hidden=network2.ReLU, act_output=network2.Softmax,dropoutpercent=0.1,regularization=network2.L2)
net1.SGD(iris_train, 10, 1, 1.0, lmbda=3.0,evaluation_data=iris_test, monitor_evaluation_cost=True,
         monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)
[22]
net1 = network2.load_network("iris-4-20-7-3.dat")
# Set hyper-parameter values individually after the network
net1.set_parameters(cost=network2.LogLikelihood, act_hidden=network2.ReLU, act_output=network2.Softmax,dropoutpercent=0.5,regularization=network2.L2)
net1.SGD(iris_train, 10, 1, 1.0, lmbda=3.0,evaluation_data=iris_test, monitor_evaluation_cost=True,
         monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)
