#n_x=Number of input features
#m is the number of training examples
import numpy as np
w=np.array([[1],
            [2],
            [3]])

b=3
#Randomly Initialise w and b



X=np.array([[2,4,6,3],
           [3,5,8,6],
           [7,9,10,11]]) #Here you can replace X with your desired matrix
           
#Create train set X which is a (n_x*m) matrix
X/=255 #flattening X
y=np.array([2,3,7,3])
#Create the outputs for the given training inputs
y/=255 #flattening Y

     



for i in range(1000): #Here 1000 is the number of iterations. More the number of iterations,More is the accuracy but also speed is compromised
    Z=np.dot(w.T,X)+b 
    
    A=1/(1+np.exp(-Z))#A is the output of a sigmoid function
    
    J = (- 1 / 4) * np.sum(y * np.log(A) + (1 - y) * (np.log(1 - A))) #Calculating cost function
    
    

    x1=np.dot(X,(A-y).T)
    dW=x1/4 #Computing derivatives
    db=np.sum(A-y) 
    db/=4
    
    w=w-0.05*dW #Gradient descent
    b=b-0.05*db
    
    print("Cost is"+str(J))
#Now the network is trained
Z_train=np.dot(w.T,X)+b 
A_train=1/(1+np.exp(-Z))
print("train accuracy: {} %".format(100 - np.mean(np.abs(A_train - y)) * 100)) #Calculate train accuracy
print("W="+str(w))
print("b="+str(b))
# Now you can use these weights in your assumptions


    
    
