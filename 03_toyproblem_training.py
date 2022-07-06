import numpy as np

def sigma(z):
    return 1 / (1 + np.exp(-z))

"""with open ('data\data_dark_bright_test_4000.csv', 'rt') as f:
    data = f.readlines()
    f.close()"""
        
class Network:
    def __init__(self, n_inp, n_hid, n_out, learning_rate):
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.learning_rate = learning_rate
        self.wA = np.random.rand(n_hid, n_inp) - 0.5
        self.wB = np.random.rand(n_out, n_hid) - 0.5
        
    def feedforward(self, ar_inp):
        #create x vector / Input
        
        x_vector = np.array(ar_inp)
        #calc hidden layer
        h = sigma(np.dot(self.wA, x_vector)) 
        
        #calc y / output
        y = sigma(np.dot(self.wB, h))
        return y, h, x_vector
    
    def test(self, data_list):
        
        
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
                x.append(element)/255
                
            y = self.feedforward(x)
            
            if y[0] > y[1]:
                guess = 0
                
            else:
                guess = 1
                
            if guess == correct_answer:
                correct += 1
            else:
                not_correct += 1

        success_rate = 100/(correct+not_correct)*correct
        print(success_rate)

    def train(self, data):
        
        
        
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
        
        #
        
        for used_line in range(len(data_list)):
            x = []
            
            # iterate through lines, get answer
            correct_answer = anatol_list[used_line][0]
            
            for element in anatol_list[used_line][1::]:
                x.append(element/255)
                
            y = self.feedforward(x)[0]
            error_out = np.array([correct_answer], dtype=object) - y
            
            error_hid = np.dot(self.wB.T, error_out)
            
            h = self.feedforward(x)[1]
            h = h.reshape(-1, 1)
            
            print(self.learning_rate * (error_out * y * (1 - y).shape))
            self.wB = self.wB + np.dot(self.learning_rate * (error_out * y * (1 - y).reshape(-1, 1)), h.T)
            self.wA = self.wA + np.dot(self.learning_rate * (error_out*h*(1 - h), self.feedforward(x)[2].T))
            
            # compare answer with guess
            if y[0] > y[1]:
                guess = 0
                
            else:
                guess = 1
                
            if guess == correct_answer:
                correct += 1
            else:
                not_correct += 1
        
        #calc and print success rate
        success_rate = 100/(correct+not_correct)*correct
        print(success_rate)

        
        
        
# oop1 = Network(784, 30, 10, 1)
# with open ('data\mnist_test.csv', 'rt') as f:
#     data_list = f.readlines()
#     f.close()
# oop1.train(data_list)
#oop1.test(data_list)
oop2 = Network(4, 3, 2, 1)
with open ('data\data_dark_bright_test_4000.csv', 'rt') as f:
    data_list = f.readlines()
    f.close()
oop2.train(data_list)
oop2.test(data_list)