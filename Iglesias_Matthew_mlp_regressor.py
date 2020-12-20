#Matthew Iglesias

import numpy as np
from utils import *
import math 
import time
from sklearn.neural_network import MLPRegressor

if __name__ == "__main__":  
    plt.close('all')
    
    #data_path = 'C:\\Users\\OFuentes\\Documents\\Research\\data\\'  # Use your own path here
    
    X = np.load('particles_X.npy').astype(np.float32)
    y = np.load('particles_y.npy').astype(np.float32)
  
    X_train, X_test, y_train, y_test = split_train_test(X,y,seed=20)
    
    model = MLPRegressor(solver='adam', alpha=1e-6, hidden_layer_sizes=(300), verbose=True, random_state=1)
    
    start = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time()-start
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    print('Training iterations  {} '.format(model.n_iter_))  
    
    start = time.time()       
    pred = model.predict(X_test)
    #accur = model.score(X_train,y_train)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))   
    print('Mean-squared error: {0:.6f}'.format(mse(y_test,pred)))
    #print('Accuracy: {0:.6f}'.format(accur))

          
   