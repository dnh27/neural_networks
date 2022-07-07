import numpy as np

def sigma(z):

    return 1 / (1 + np.exp(-z))

"""with open ('data\data_dark_bright_test_4000.csv', 'rt') as f:
    data = f.readlines()
    f.close()"""
class Network:
    def Store(self, file):
        arrays = [layer.weights for layer in self.layers]
        np.savez(file, arrays)
    def __init__(self, n_inp, n_hid, n_out, learning_rate):
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.learning_rate = learning_rate
        self.w_a = np.random.rand(n_hid, n_inp) - 0.5
        self.w_b = np.random.rand(n_out, n_hid) - 0.5
        
    def feedforward(self, ar_inp):
        #create x vector / Input
        
        x_vector = np.array(ar_inp).reshape(-1,1)
        #calc hidden layer
        h = sigma(np.dot(self.w_a, x_vector)) 
        
        #calc y / output
        y = sigma(np.dot(self.w_b, h))
        return y, h, x_vector
    
    def success_ratio(answer, y):
        pass
    
    def test(self, data_list, data_size):        
        #Edit data
        
        anatol_list = []
        
        for line in data_list:              
            line = line.strip("\n")
            line = line.split(",")
            num_list = []
            
            for num in line:
                num = int(num)
                num_list.append(num)
                
            anatol_list.append(num_list)
        
        correct = 0
        not_correct = 0
        
        for used_line in range(len(data_list)):
            x = []
            correct_answer = anatol_list[used_line][0]
            
            for element in anatol_list[used_line][1::]:
                
                x.append(element/data_size)
                
            y, _, x_vector = self.feedforward(x)
            
            if correct_answer == np.argmax(y):
                correct += 1

            
            #print("y=", y, "\n", "s=", correct_answer)
            # ideal = np.zeros(x_vector, 1)
            # if correct_answer == np.argmax(y):
            
            # if y[0] > y[1]:
            #     guess = 0
            #     # print("y[0]=", y[0])
                
            # else:
            #     guess = 1
                
            # if guess == correct_answer:
            #     correct += 1
            # else:
            #     not_correct += 1
        
            
        success_rate = 100/len(data_list)*correct
        
        print("success rate =", success_rate)
        
            
        
    def onehot(self, solution, length):
        zeros = np.zeros((length, 1))
        for i in range(length):
            if i == solution:
                zeros[i][0] = 1
                
                break
        
        return zeros
        
    def train(self, data, data_size): 
        #Edit data
        #print("random weights=", "\n", self.w_a, "\n", "\n", self.w_b )
        data_list = []
        #print("train")
        for line in data:              
            line = line.strip("\n")
            line = line.split(",")
            num_list = []
            
            for num in line:
                num = int(num)
                num_list.append(num)
                
            data_list.append(num_list)
        for used_line in range(len(data_list)):
            x = []
            
            # iterate through lines, get answer
            correct_answer = data_list[used_line][0]
            
            for element in data_list[used_line][1::]:
                x.append(element/data_size)
                
            y, h, x_vector = self.feedforward(x)
            y = y.reshape(-1, 1)
            h = h.reshape(-1, 1)
            x_vector = x_vector.reshape(-1, 1)
            
            error_out = self.onehot(correct_answer, y.shape[0]) - y
            error_hid = np.dot(self.w_b.T, error_out)
            
            self.w_b = self.w_b + self.learning_rate * np.dot((error_out * y * (1 - y)), h.T)
            self.w_a = self.w_a + self.learning_rate * np.dot((error_hid * h * (1 - h)), x_vector.T)
            print("training...", int(100/len(data)*used_line), "%")
        # print(self.w_a.shape, self.w_b.shape)
        # print(self.w_a,"\n", "\n",   self.w_b)
        
        #calc and print success rate
        
        
        
# oop1 = Network(784, 30, 10, 1)
# with open ('data\mnist_test.csv', 'rt') as f:
#     data_list = f.readlines()
#     f.close()
# oop1.train(data_list)
#oop1.test(data_list)
oop2 = Network(784, 80, 10, 0.01)
with open ('data\mnist_train.csv', 'rt') as f:
    raw_data = f.readlines()
    f.close()
with open ('data\mnist_test.csv', 'rt') as f:
    test_data = f.readlines()
    f.close
oop2.train(raw_data, 784)
oop2.test(test_data, 784)

