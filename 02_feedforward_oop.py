import numpy as np

def sigma(z):
    return 1 / (1 + np.exp(-z))

"""with open ('data\data_dark_bright_test_4000.csv', 'rt') as f:
    data = f.readlines()
    f.close()"""
        
class Network:
    def __init__(self, n_inp, n_hid, n_out):
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.wA = np.array([[-0.3, -0.7, -0.9, -0.9], [-1, -0.6, -0.6, -0.6], [0.8, 0.5, 0.7, 0.8]])
        self.wB = np.array([[2.6, 2.1, -1.2], [-2.3, -2.3, 1.1]])

        #self.wA = np.random.rand(n_hid, n_inp) - 0.5
        #self.wB = np.random.rand(n_out, n_hid) - 0.5
        
    def feedforward(self, ar_inp, data_size=1):
     
        x_vector = np.array(ar_inp)/data_size
       
        h = sigma(np.dot(self.wA, x_vector)) 
        
        y = sigma(np.dot(self.wB, h))
            
        return y
    
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
                x.append(element)  
                
            y = self.feedforward(x, 255)
            
            if y[0] > y[1]:
                guess = 0
                
            else:
                guess = 1
                
            
            print(guess, correct_answer)
            
            
            if guess == correct_answer:
                correct += 1
            else:
                not_correct += 1

        
        

        

        
        print(100/(correct+not_correct)*correct)

oop = Network(4, 3, 2)
with open ('data\data_dark_bright_test_4000.csv', 'rt') as f:
    data_list = f.readlines()
    f.close()
oop.test(data_list)