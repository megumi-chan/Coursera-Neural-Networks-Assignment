import numpy as np
import scipy.io
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('./Datasets/dataset3.mat')
neg_set = mat['neg_examples_nobias']
pos_set = mat['pos_examples_nobias']
init_w = np.transpose(mat['w_init'])
feas_w = np.transpose(mat['w_gen_feas'])

#Helper function to draw
def fun(x, y):
    return (0 - init_w[0][0]*x - init_w[0][1]*y)/init_w[0][2]

def fun2d(x):
    return (0 - init_w[0][0]*x)/init_w[0][1]

# Plot the dataset
def plot():
    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    ax2 = fig.add_subplot(212)
    x = y = np.arange(-3.0, 3.0, 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z)
    for train_set in neg_set:
        ax.scatter(train_set[0], train_set[1], train_set[2], c='r', marker='o')
        ax2.scatter(train_set[0], train_set[1], c='r', marker='o')
    for train_set in pos_set:
        ax.scatter(train_set[0], train_set[1], train_set[2], c='b', marker='^')
        ax2.scatter(train_set[0], train_set[1], c='b', marker='^')
    xs = np.arange(-1, 1, 0.05)
    ax.view_init(45, 180)
    ax2.plot(xs, [fun2d(i) for i in xs])
    plt.show()
    
def classified(train_set):
    return np.dot(init_w[0], train_set) >= 0
    
def num_of_errors():
    error = 0
    for train_set in neg_set:
        if np.dot(init_w[0], train_set) >= 0:
            error += 1
    for train_set in pos_set:
        if np.dot(init_w[0], train_set) < 0:
            error += 1        
    return error

# Add bias component
neg_set = np.append(neg_set, np.ones((np.shape(neg_set)[0],1), dtype = np.int64), axis = 1)
pos_set = np.append(pos_set, np.ones((np.shape(pos_set)[0],1), dtype = np.int64), axis = 1)
step = 0
#Train
for i in range(20):
    for train_set in neg_set:
        if classified(train_set):
            init_w = init_w - train_set
            plot()
            print(num_of_errors())
            
    for train_set in pos_set:
        if not classified(train_set):
            init_w = init_w + train_set
            plot()
            print(num_of_errors())
    step += 1
    if num_of_errors() == 0:
        break;

print(init_w)
print("Total of steps:" + str(step))

