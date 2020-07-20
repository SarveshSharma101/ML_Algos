#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import h5py


# In[3]:


f = h5py.File('train_catvnoncat.h5')


# In[4]:


print(list(f))


# In[4]:


train_y = np.array(f['train_set_y'])
train_x = np.array(f['train_set_x'])
classes = np.array(f['list_classes'])


# In[5]:


print("Y shape", train_y.shape)
print("X shape", train_x.shape)
print("classes shape", classes.shape)
print("classes",classes[0],classes[1])


# In[6]:


train_x_f = (train_x.reshape(209,-1).T)/255
train_x_f.shape
train_x_f[:,1]


# In[7]:


def imshow(num):
    
    for i in range(1,num+1):
        plt.subplot(5,5,i)
        plt.imshow(train_x[i-1])
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("y ="+str(train_y[i-1])+" it is a "+str(classes[train_y[i-1]]))

plt.figure(figsize=(10,10))
imshow(10)
plt.show()


# In[8]:


def sigmoid(z):
    
    s = 1/(1+np.exp(-z))
    return s


# In[9]:


def initalize_parameter(dim):
    
    w = np.zeros(shape=(dim,1))
    b = 0
    
    return w,b
w1 , b1 = initalize_parameter(2)
print(w1,b1)


# In[10]:


def propogate(X,w,b,y):
    
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)
    cost = (-1/m)*np.sum(y*np.log(A)+(1-y)*np.log(1-A))
    
    dz= (A-y)
    dw = (1/m)*np.dot(X, dz.T)
    db =np.sum(dz)
    
    grads = {"dw": dw,
            "db":db}
    
    return grads, cost

    
    


# In[11]:


def optimize(X,w,b,y,num_iteration, learning_rate, print_cost):
    m = X.shape[1]
    costs=[]
    
    for i in range(num_iteration):
        grads, cost = propogate(X,w,b,y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w- learning_rate*dw
        b = b- learning_rate*db
        
        if print_cost and (i%1000 ==0):
            print(i,"th cost is",cost)
            costs.append(cost)
    parameter={"w": w,
              "b":b}
    return parameter, costs
        


# In[12]:


def predict(X,w,b):
    
    A = sigmoid(np.dot(w.T, X)+b)
    m = X.shape[1]
    y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(w.T, X) +b)
    
    for i in range(A.shape[1]):
        if A[0,i]>0.5:
            y_prediction[0,i]=1
        else:
            y_prediction[0,i]=0
    return y_prediction
            


# In[13]:


def model(X ,y, num_iterations=2000, learning_rate= 0.5,print_cost=False):
    
    w,b = initalize_parameter(X.shape[0])
    
    params, grade = optimize(X,w,b,y,num_iterations, learning_rate=learning_rate,print_cost=print_cost)
    
    w = params["w"]
    b = params["b"]
    y_pred_train= predict(X,w,b)
    print("---------->", y_pred_train)
    print(y)
    print("training_accuracy:{}".format(100-np.mean((y_pred_train-y))*100))
    
    return w,b


# In[14]:


w,b=model(train_x_f, train_y, 27000,0.005, True) 


# In[15]:


import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y.ravel(), cmap=plt.cm.Spectral)
    

def load_planar_dataset():
    np.random.seed(1)
    m = 400 
    N = int(m/2) 
    D = 2 
    X = np.zeros((m,D)) 
    Y = np.zeros((m,1), dtype='uint8') # vector de etiquetas (0 para rojo, y 1 para azul)
    a = 4 # trazado maximo de la flor

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radio
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T

    return X, Y

def load_extra_datasets():  
    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)
    
    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure


# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
X, y =load_planar_dataset()

print("X",X.shape)
print("y",y.shape)


# In[21]:


#Visualizing datasets
plt.scatter(X[0,:], X[1,:],c= y.ravel(),s=20,cmap=plt.cm.Spectral)


# In[18]:


print("Shape of X->",X.shape)
print("Shape of y->",y.shape)
print("Number of training examples->",y.shape[1])


# In[19]:


pw , pb = initalize_parameter(2)


# In[24]:


pw, pb = model(X,y,27000,0.005, True)


# In[25]:


print("X shape", X.shape)
print("Y shape", y.shape)
print("Number of instance", X.shape[1])


# In[34]:


def layer_nodes_n(X, y):
    n_in = X.shape[0]
    n_h = 4
    n_out = y.shape[0]
    
    return (n_in, n_h, n_out)


# In[49]:


def initialize_parameters(n_in, n_h, n_out):
    
    W1 = np.random.randn(n_h, n_in)
    b1 = np.zeros(shape=(n_h,1))
    W2 = np.random.randn(n_out, n_h)
    b2 = np.zeros(shape=(n_out,1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
    


# In[50]:


def forward_propogation(X, parameters):
    W1= parameters["W1"]
    b1= parameters["b1"]
    W2= parameters["W2"]
    b2= parameters["b2"]
    
    Z1 = np.dot(W1,X)+b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


# In[89]:


def compute_cost(A,y):
    
    m = y.shape[1]
    cost = (-1/m)*np.sum(y*np.log(A) + (1-y)*np.log(1-A))
    return cost


# In[90]:


def back_prop(X, cache, y,  parameters):
    
    m = X.shape[1]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    Z2 = cache["Z2"]
    
    dZ2 = (A2-y)
    dW2 = (1/m)*np.dot(dZ2,A1.T)
    db2 = (1/m)*np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1/m)*np.dot(dZ1, X.T)
    db1 = (1/m)*np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads


# In[91]:


def update_param(params, grads, learning_rate):
    W1 = params["W1"]
    W2 = params["W2"]
    b1 = params["b1"]
    b2 = params["b2"]
    
    dW1 = grads["dW1"]
    dW2 = grads["dW2"]
    db1 = grads["db1"]
    db2 = grads["db2"]
    
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
        
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


# In[103]:


def nn_model(X,y,n_h, num_i, learning_rate, print_cost):
    m = X.shape[1]
    n_in, n_h1, n_out= layer_nodes_n(X,y)
    n_h1 = n_h
    parameters = initialize_parameters(n_in, n_h1, n_out)
    costs=[]
    for i in range(num_i):
        A2, cache = forward_propogation(X, parameters)
        cost = compute_cost(A2,y)
        
        grads  =back_prop(X, cache, y, parameters)
        
        parameters = update_param(parameters, grads, learning_rate)
        
        if (i%1000==0) and print_cost:
            print(i,"th cost is ", cost)
            costs.append(cost)
    return parameters
        
        
        
    


# In[104]:


def predict(parameters, X):
   
    A2, cache = forward_propogation(X, parameters)
    predictions = np.round(A2)

    
    return predictions


# In[106]:


parameters = nn_model(X, y, 4,num_i=10000,learning_rate=1.2, print_cost=True)


# In[107]:


plot_decision_boundary(lambda x: predict(parameters, x.T), X, y)
plt.title("Decision Boundary for hidden layer size " + str(4))


# In[108]:



# Print accuracy
predictions = predict(parameters, X)
print ('Accuracy: %d' % float((np.dot(y, predictions.T) + np.dot(1 - y, 1 - predictions.T)) / float(y.size) * 100) + '%')


# In[111]:


plt.figure(figsize=(10, 15))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 3, i + 1)
    plt.title('Hidden Layer of size %d' % n_h)
    plt.yticks([])
    plt.xticks([])
    parameters = nn_model(X, y,n_h, num_i=5000,learning_rate=1.2, print_cost=False)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(y, predictions.T) + np.dot(1 - y, 1 - predictions.T)) / float(y.size) * 100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))


# In[ ]:




