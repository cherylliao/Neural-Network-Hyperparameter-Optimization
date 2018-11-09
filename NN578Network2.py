"""
Cheryl Liao
CSC578-701 
Homework #2: Backprop Hyper-parameters
###########
10/9 Updated

10/2018 
NN578_network2.py
==============

Modified from the NNDL book code "network2.py" to be 
compatible with Python 3.

Also from "network2.py", the function ("load(filename)") is
added to this file, renamed as "load_network(filename)".
This function loads a saved network encoded in a json file.

"""

"""network2.py
~~~~~~~~~~~~~~

An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights.  Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.

"""

#### Libraries
# Standard library
import json
import random
import sys

# Third-party libraries
import numpy as np
from scipy.optimize import minimize
import scipy.stats as stats
import time

#### 10/2018 nt:
#### Definitions of the cost functions (as function classes)
class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output ``y``.
        """
        return 0.5*np.linalg.norm(y-a)**2
    ## 10/2018 nt: addition
    @staticmethod
    def derivative(a, y):
        """Return the first derivative of the function."""
        return -(y-a)

class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def derivative(a, y):
        """Return the first derivative of the function."""
        ###
        return -((a-y)/(a-1)*a)
        ###

class LogLikelihood(object):
    
    @staticmethod
    def fn(a,y):
        a=Softmax.fn(a)
        return -np.sum(stats.norm.logpdf(a,y))
    
    @staticmethod
    def derivative(a, y):
        """Return the first derivative of the function."""
        ###
        a=Softmax.fn(a)
        return -(1/(stats.norm.logpdf(a,y)))
###
    
#### 9/2018 nt:
#### Definitions of the activation functions (as function classes)
class Sigmoid(object):
    
    @staticmethod
    def fn(z):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-z))

    @classmethod
    def derivative(cls,z):
        """Derivative of the sigmoid function."""
        return cls.fn(z)*(1-cls.fn(z))
    def undo(z):
        return z*(1.0+np.exp(-z))

class Softmax(object):
    @staticmethod
    # Parameter z is an array of shape (len(z), 1).
    def fn(z):
        """The softmax of vector z."""
        ###
        e_x = np.exp(z - np.max(z))
        return e_x / e_x.sum(axis=0)
        ###

    @classmethod
    def derivative(cls,z):
        """Derivative of the softmax.  
        IMPORTANT: The derivative is an N*N matrix.
        """
        a = cls.fn(z) # obtain the softmax vector
        return np.diagflat(a) - np.dot(a, a.T)
class Tanh(object):
    @staticmethod
    # Parameter z is an array of shape (len(z), 1).
    def fn(z):
      
        
        return np.tanh(z)
    ###
    
    @classmethod
    def derivative(cls,z):
       
        
        return 1-cls.fn(z)**2
    def undo(z):
        return z*(1/np.tanh(z))
#This is LEAKY ReLU!!!
class ReLU(object):
    @staticmethod
    # Parameter z is an array of shape (len(z), 1).
    def fn(z):
        return np.log(1+np.exp(z))
    ###
    
    @classmethod
    def derivative(cls,z):
        a=cls.fn(z)
        return [1.0/(1.0+np.exp(-z)) for z in a]
    def undo(z):
        return z*(1/np.log(1+np.exp(z)))
     
   
#we are not dividing any of these by n?
class L2(object):
    @staticmethod
    # Parameter c is the cost function
    #w is the weight vector
    def fn(w,lmbda):
        epsilon=1e-12
        output=0.5*lmbda*np.sum(np.square(w))
        return np.clip(output,epsilon, 1. - epsilon)
    ###
class L1(object):
    
    @staticmethod
    def fn(w,lmbda):
        epsilon=1e-12
        output=lmbda*np.sum(np.linalg.norm(w))
        return np.clip(output,epsilon, 1. - epsilon)
#### Main Network class
class Network(object):

    ## 10/2018 nt: additional keyword arguments for hyper-parameters
    def __init__(self, sizes, cost=CrossEntropyCost, act_hidden=Sigmoid, \
                 act_output=None, regularization=None, dropoutpercent=0.0):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.dropoutpercent = dropoutpercent
        #to access the original hidden layers
        self.oldLayers=self.sizes[1:-1]
   

        self.default_weight_initializer()
        self.cost=cost
        
        ## 10/2018 nt: addition
        self.act_hidden = act_hidden
        if act_output == None:
            self.act_output = self.act_hidden
        else:
            self.act_output = act_output
        self.regularization = regularization
       

    ## 10/2018 nt: convenience function for setting hyper-parameters
    def set_parameters(self, cost=QuadraticCost, act_hidden=Sigmoid, \
                       act_output=None, regularization=None, dropoutpercent=0.0):
        self.dropoutpercent = dropoutpercent
        
        
        self.cost=cost
        self.act_hidden = act_hidden
        if act_output == None:
            self.act_output = self.act_hidden
        else:
            self.act_output = act_output
        self.regularization = regularization
        
    #dropout function called to drop nodes
    def dropout(self):
        
        p=self.dropoutpercent
        bs=len(self.biases[1:-1])
        ws=len(self.weights[1:-1])
        repeat=ws/bs
        u1 = np.random.binomial(1, p, size=bs) / p
        #if the bias is 0, the weights are all zeros
        u2 = np.repeat(u1,repeat)
        self.biases[1:-1]*=u1
        self.weights[1:-1]*=u2
      

            
    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        #first set hidden layer = output no matter what
        #self.act_output = self.act_hidden
        
        for b, w in zip(self.biases, self.weights):
                    
            ## 10/2018: THIS NEEDS CHANGE. The previous line (commented) 
            ##  doesn'twork any more.  The new scheme is written which 
            ##  utilizes function class.  But this works but is still 
            ##  incorrect because you have to consider act_output as well.
            #a = sigmoid(np.dot(w, a)+b)
     
            a = (self.act_hidden).fn(np.dot(w, a)+b)
     
                
        #what to do if there is diff. activation function for
        #output, undo the last activation function then reapply
        #correct one
        if self.act_output!=None:
            a=(self.act_hidden).undo(a) 
            a=(self.act_output).fn(a) 
               #normalize if Tanh is used and cost is not quadratic    
        if self.act_hidden==Tanh and self.cost!=QuadraticCost:
            normalize(a)
            
        return a
        

    ## 9/2018 nt: additional parameter 'no_convert' to control the vectorization of the target.
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            no_convert=True):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.

        """
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):#xrange(epochs):
            #random.shuffle(training_data)
         
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]#xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
                
            print ("Epoch %s training complete" % j)
            if self.dropoutpercent>0:
                self.dropout()
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda) # nt: for cost, always NO convert (default) for training
                training_cost.append(cost)
                print ("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True) # nt: for accuracy, always _DO_ convert (argmax) for training
                training_accuracy.append(accuracy)
                print ("Accuracy on training data: {} / {}".format(
                    accuracy, n))
            if monitor_evaluation_cost:
                ## 9/2018 nt: changed the last parameter convert
                if no_convert:
                    cost = self.total_cost(evaluation_data, lmbda) # nt: if test/val data is already vectorized for y
                else:
                    cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print ("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                ## 9/2018 nt: changed the last parameter convert
                if no_convert:
                    accuracy = self.accuracy(evaluation_data, convert=True) #nt: _DO_ convert (argmax)
                else:
                    accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
    
                print ("Accuracy on evaluation data: {} / {}".format(
                    ## 9/2018 nt: This seems like a bug!
                    #self.accuracy(evaluation_data), n_data))
                    accuracy, n_data))
            print ('')
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy
        
    ## 10/2018: THIS NEEDS CHANGE to incorporate self.regularization. 
    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.

        """
      
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    
        if self.regularization==None:
            lmbda=0
            self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        else:
            
            self.weights = [(w-eta*self.regularization.fn(w,lmbda))-(eta/len(mini_batch))*nw
                       for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        if self.dropoutpercent>0:
            self.default_weight_initializer()


    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            ## 9/2018 nt: changed
            #activation = sigmoid(z)
            activation = (self.act_hidden).fn(z)
            activations.append(activation)

        # backward pass
        ## 9/2018 nt: Cost and activation functions are parameterized now.
        ##            Call the activation function of the output layer with z.
        #delta = (self.cost).delta(zs[-1], activations[-1], y)
        a_prime = (self.act_output).derivative(zs[-1]) # 9/2018 nt: changed, da/dz
        #delta = (self.cost).derivative(activations[-1], y) * a_prime # 9/2018 nt: changed, dC/da * da/dz
        c_prime = (self.cost).derivative(activations[-1], y) #10/2018 nt: split a line to accommodate Softmax
        if self.act_output == Softmax:
            delta = np.dot(a_prime, c_prime)
        else:
            delta = c_prime * a_prime # 9/2018 nt: changed, dC/da * da/dz

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
 
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        #!!!loop through hidden layers
        for l in range(2, self.num_layers):#xrange(2, self.num_layers):
            z = zs[-l]
            ## 9/2018 nt: Changed to call the activation function of the hidden layer with z.
            #sp = sigmoid_prime(z)
            sp = (self.act_hidden).derivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        if self.dropoutpercent>0:
            
            self.default_weight_initializer()
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.

        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)
    
    ## 10/2018: THIS NEEDS CHANGE to incorporate self.regularization. 
    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            #accessing the method of however cost was set to
     
            cost += self.cost.fn(a, y)/len(data)
        if self.regularization==None:
            lmbda=0
            cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
            return cost
        else:
            cost+=[self.regularization.fn(w,lmbda)/len(data) for w in self.weights]
            return cost[0]
    

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Loading a Network from a json file
def load_network(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    #cost = getattr(sys.modules[__name__], data["cost"])
    #net = Network(data["sizes"], cost=cost)
    net = Network(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Miscellaneous functions
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
