import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer


def sigmoid(x):
    """ Sigmoid function.
    This function accepts any shape of np.ndarray object 
    as input and perform sigmoid operation.
    """
    return 1 / (1 + np.exp(-x))

def grad_sigmoid(y):
    """ First derivative of Sigmoid function.
    The input to this function should be the value that 
    output from sigmoid function.
    """
    return y * (1 - y)


class GenData:
    @staticmethod
    def _gen_linear(n=100):
        """ Data generation (Linear)

        Args:
            n (int):    the number of data points generated in total.

        Returns:
            data (np.ndarray, np.float):    the generated data with shape (n, 2). Each row represents
                a data point in 2d space.
            labels (np.ndarray, np.int):    the labels that correspond to the data with shape (n, 1).
                Each row represents a corresponding label (0 or 1).
        """
        data = np.random.uniform(0, 1, (n, 2))

        inputs = []
        labels = []

        for point in data:
            inputs.append([point[0], point[1]])

            if point[0] > point[1]:
                labels.append(0)
            else:
                labels.append(1)

        return np.array(inputs), np.array(labels).reshape((-1, 1))

    @staticmethod
    def _gen_xor(n=100):
        """ Data generation (XOR)

        Args:
            n (int):    the number of data points generated in total.

        Returns:
            data (np.ndarray, np.float):    the generated data with shape (n, 2). Each row represents
                a data point in 2d space.
            labels (np.ndarray, np.int):    the labels that correspond to the data with shape (n, 1).
                Each row represents a corresponding label (0 or 1).
        """
        data_x = np.linspace(0, 1, n // 2)

        inputs = []
        labels = []

        for x in data_x:
            inputs.append([x, x])
            labels.append(0)

            if x == 1 - x:
                continue

            inputs.append([x, 1 - x])
            labels.append(1)

        return np.array(inputs), np.array(labels).reshape((-1, 1))

    @staticmethod
    def fetch_data(mode, n):
        """ Data gather interface

        Args:
            mode (str): 'Linear' or 'XOR', indicate which generator is used.
            n (int):    the number of data points generated in total.
        """
        assert mode == 'Linear' or mode == 'XOR'

        data_gen_func = {
            'Linear': GenData._gen_linear,
            'XOR': GenData._gen_xor
        }[mode]

        return data_gen_func(n)


class SimpleNet:
    def __init__(self,input_size=2, hidden1_size=256, hidden2_size=128, output_size=1, lr=0.005, num_step=2000, print_interval=100):
        """ A hand-crafted implementation of simple network.

        Args:
            hidden_size:    the number of hidden neurons used in this model.
            num_step (optional):    the total number of training steps.
            print_interval (optional):  the number of steps between each reported number.
        """
        self.num_step = num_step
        self.print_interval = print_interval
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.learn_rate = lr

        # Model parameters initialization
        # Please initiate your network parameters here.
        self.hidden_weights = {
            'W1': np.random.normal(loc=0, scale=1, size=(input_size, hidden1_size)),
            'W2': np.random.normal(loc=0, scale=1, size=(hidden1_size, hidden2_size)),
            'W3': np.random.normal(loc=0, scale=1, size=(hidden2_size, output_size))
        }
        
        self.hidden_biases = {
            'B1': np.random.normal(loc=0, scale=1, size=(hidden1_size)),
            'B2': np.random.normal(loc=0, scale=1, size=(hidden2_size)),
            'B3': np.random.normal(loc=0, scale=1, size=(output_size))
        }

    @staticmethod
    def plot_result(data, gt_y, pred_y):
        """ Data visualization with ground truth and predicted data comparison. There are two plots
        for them and each of them use different colors to differentiate the data with different labels.

        Args:
            data:   the input data
            gt_y:   ground truth to the data
            pred_y: predicted results to the data
        """
        assert data.shape[0] == gt_y.shape[0]
        assert data.shape[0] == pred_y.shape[0]

        plt.figure()

        plt.subplot(1, 2, 1)
        plt.title('Ground Truth', fontsize=18)

        for idx in range(data.shape[0]):
            if gt_y[idx] == 0:
                plt.plot(data[idx][0], data[idx][1], 'ro')
            else:
                plt.plot(data[idx][0], data[idx][1], 'bo')

        plt.subplot(1, 2, 2)
        plt.title('Prediction', fontsize=18)

        for idx in range(data.shape[0]):
            if pred_y[idx] == 0:
                plt.plot(data[idx][0], data[idx][1], 'ro')
            else:
                plt.plot(data[idx][0], data[idx][1], 'bo')

        plt.show()


    def forward(self, inputs):
        """ Implementation of the forward pass."""
        ##### 1st hidden layer
        Z1 = np.dot(inputs, self.hidden_weights['W1']) + self.hidden_biases['B1']
        self.X1 = sigmoid(Z1)
        ##### 2nd hidden layer
        Z2 = np.dot(self.X1, self.hidden_weights['W2']) + self.hidden_biases['B2']
        self.X2 = sigmoid(Z2)
        ##### output layer
        Z3 = np.dot(self.X2, self.hidden_weights['W3']) + self.hidden_biases['B3']
        NNpred = sigmoid(Z3)
        return NNpred


    def backward(self, inputs, grad_Loss, NNpred):
        """ Implementation of the backward pass. """
        ########### Calculate Gradients ###########
        ##### output layer
        grad_Z3 = grad_Loss*grad_sigmoid(NNpred)
        grad_W3 = grad_Z3*(self.X2).T
        grad_B3 = grad_Z3.squeeze()
        
        ##### 2nd hidden layer
        grad_Z2 = np.dot(grad_Z3, self.hidden_weights['W3'].T)*grad_sigmoid(self.X2)
        grad_W2 = np.dot(self.X1.T, grad_Z2)
        grad_B2 = grad_Z2.squeeze()
        
        ##### 1st hidden layer
        grad_Z1 = np.dot(grad_Z2, self.hidden_weights['W2'].T)*grad_sigmoid(self.X1)
        grad_W1 = np.dot(inputs.T, grad_Z1)
        grad_B1 = grad_Z1.squeeze()

        ########### Update Model Parameters ###########
        self.hidden_weights['W1'] -= self.learn_rate*grad_W1
        self.hidden_weights['W2'] -= self.learn_rate*grad_W2
        self.hidden_weights['W3'] -= self.learn_rate*grad_W3
        self.hidden_biases['B1'] -= self.learn_rate*grad_B1
        self.hidden_biases['B2'] -= self.learn_rate*grad_B2
        self.hidden_biases['B3'] -= self.learn_rate*grad_B3


    def train(self, inputs, labels):
        """ The training routine that runs and update the model.

        Args:
            inputs: the training (and testing) data used in the model.
            labels: the ground truth of correspond to input data.
        """
        # make sure that the amount of data and label is match
        assert inputs.shape[0] == labels.shape[0]

        n = inputs.shape[0]
        
        acc_his = []
        for epochs in range(self.num_step):
            for idx in range(n):
                # operation in each training step:
                #   1. forward passing
                #   2. compute loss 
                #   3. propagate gradient backward to the front
                NNpred = self.forward(inputs[idx:idx+1, :])
                grad_Loss = 2*(NNpred - labels[idx:idx+1, :])  #loss的gradient
                self.backward(inputs[idx:idx+1, :], grad_Loss, NNpred)

            if epochs % self.print_interval == 0:
                self.test(inputs, labels, epochs)
            
#            if epochs % 100 == 0:
#                acc100 = self.test(inputs, labels, epochs)
#                acc_his.append(acc100)

        print('***** Training finished *****')
        self.test(inputs, labels, epochs+1)
        
        return np.array(acc_his)


    def test(self, inputs, labels, epoch):
        """ The testing routine that run forward pass and report the accuracy.

        Args:
            inputs: the testing data. One or several data samples are both okay.
                The shape is expected to be [BatchSize, 2].
            labels: the ground truth correspond to the inputs.
        """
        n = inputs.shape[0]

        error = 0.0
        loss = 0.0
        for idx in range(n):
            result = self.forward(inputs[idx:idx+1, :])
            loss += (result - labels[idx:idx+1, :])**2
            error += abs(result - labels[idx:idx+1, :])

        error /= n
        acc = (1 - error)
        acc = acc[0,0]
        loss = loss[0,0]
        print(f'Epoch {epoch:4d}, loss: {loss:.2f}, accuracy: {acc:.2%}')
        
        return acc


if __name__ == '__main__':
    ######## Set parameters #######
    data_group = 'XOR'  #'XOR', 'Linear'
    num_step = 3000
    print_interval = 500
    learn_rate = 0.001
    alpha1 = 2**0
    alpha2 = 2**0
    
    ######## Generate Data #########
    data, label = GenData.fetch_data(data_group, 70)  #'XOR'
    
#    #### 預先跑好的Linear data points (for 固定分佈)
#    data = pd.read_csv('Linear_data.csv')#,header=None)
#    label = pd.read_csv('Linear_label.csv')
#    data = data.values
#    label = label.values
    
    ######## Create Model object #########
    net = SimpleNet(num_step=num_step, print_interval=print_interval,
                    hidden1_size=64*alpha1, hidden2_size=64*alpha2, lr=learn_rate)
    
    ######## Start Model Training #########
    start_time = timer()
    acc_his = net.train(data, label)
    end_time = timer()
    print("Model Training time taken: {0} minutes {1:.1f} seconds".format((end_time - start_time)//60, (end_time - start_time)%60))

#    # save acc history (per 100 epoch)
#    acc_his = pd.DataFrame(acc_his)
#    acc_fn = data_group +'_hidden' + str(64*alpha1) + '-' + str(64*alpha2)+ '_lr' +str(learn_rate) + '_Epoch' + str(num_step) 
#    acc_his.to_csv(acc_fn +'.csv', index=False)

    ######## Trained Model Inference and Plot #########
    ##### Training data
    pred_prob = net.forward(data)
    pred_result = np.round(pred_prob)
#    print('\nSimpleNet Predicted Probabilities:')
#    print(pred_prob)
    SimpleNet.plot_result(data, label, pred_result)
    
    ##### Test data
#    test_data, test_label = GenData.fetch_data(data_group, 100)  #'XOR'
    test_data, test_label = GenData.fetch_data('XOR', 100)  #'XOR'
    pred_prob_test = net.forward(test_data)
    pred_result_test = np.round(pred_prob_test)
#    print('\nSimpleNet Predicted Probabilities (Test data):')
#    print(pred_prob_test)
    print('\n\n***** Predicted results of Test data *****')
    net.test(test_data, test_label, num_step)
    SimpleNet.plot_result(test_data, test_label, pred_result_test)

#################################################################
########### Other code for different purpose ############
#import matplotlib.pyplot as plt


#import numpy as np
#data, label = GenData.fetch_data('Linear', 70)  #'XOR', 'Linear'
#import pandas as pd
##Linear_data_pts = np.hstack([data, label])
#
#Linear_data = pd.DataFrame(data, columns=["dataX","dataY"])
#Linear_label = pd.DataFrame(label, columns=["label"])
#Linear_data.to_csv('Linear_data.csv', index=False)
#Linear_label.to_csv('Linear_label.csv', index=False)