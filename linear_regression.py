import numpy as np

class LinearRegressionMGD():
    
    def __init__(self):
        self.degree = 0
        self.m = 0
        self.b = 0
        self.costs = []
    
    def cost(self, x, y):
        cost = 0
        for i in range(len(x)):
            cost = cost + (self.m.dot(x[i])+self.b-y[i])**2
        return cost
    
    
    def grad(self, x, y):
        sum_m = np.array([0 for i in range(self.degree)], dtype=float)
        sum_b = 0
        
        
        for i in range(len(x)):
            sum_m+= 2 * x[i] * ( self.m.dot( x[i] ) + self.b - y[i] )
            sum_b+= 2 * ( self.m.dot( x[i] ) + self.b - y[i] )
        
        return sum_m,sum_b
    
    
    def fit(self,x,y,lr=0.001, batch_size=10,epochs=200, threshold=0.00001, show_epochs=1):

        if self.degree==0:
            self.degree=len(x[0])
            self.m=np.array([0 for i in range(self.degree)])

        step_m=np.array([1.0 for i in range(self.degree)])
        step_b=1.0

        if batch_size>len(x):
            batch_size=len(x)

        epoch=0
        while ((max(abs(step_m))>threshold) | abs((step_b)>threshold )) & (epoch<epochs):
            index=0
            a=np.arange(len(x))
            np.random.shuffle(a)
            x=x[a]
            y=y[a]
            avg_step_m=np.array([1.0 for i in range(self.degree)])
            avg_step_b=1.0
            
            
            epoch+=1
            

            for j in range(batch_size-1, len(x),batch_size):
                

                mini_batch_x = x[index:j+1,:]
                mini_batch_y = y[index:j+1]
                
                index+=batch_size
            

                step_m,step_b=self.grad(mini_batch_x,mini_batch_y)

                


                if(np.isnan(step_m.sum()) | np.isnan(step_b)):
                    print('Nan Value Encountered! Try reducing Learning Rate by a factor of 10')
                    return


                self.m = self.m - lr / batch_size * step_m
                self.b = self.b - lr / batch_size * step_b
                
                avg_step_m+= step_m
                avg_step_b+= step_b
                

            self.costs.append(self.cost(x,y))
            
            if epoch%show_epochs==0:
                print(f' Epoch: {epoch} Cost: {self.costs[-1]} ')
        
    
    def predict(self,x):
        

        
        pred=[]
        
        for i in range(len(x)):
            pred.append(self.m.dot(x[i]) + self.b)
        
        return pred