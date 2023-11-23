import numpy as np

class rmsprop:
    def __init__(self, beta=0.9, w_shape, b_shape) -> None:
        self.beta = beta
        self.sdw = np.zeros(w_shape)
        self.sdb = np.zeros(b_shape)

    
    def get_dw_opt(self, dw):
        self.sdw = self.beta*self.sdw + (1-self.beta)*dw**2
        return dw/np.sqrt(self.sdw)
    

    def get_db_opt(self, db):
        self.sdb = self.beta*self.sdb + (1-self.beta)*db**2
        return db/np.sqrt(self.sdb)
    

class adam:
    def __init__(self, beta1=0.9, beta2=0.999, w_shape, b_shape) -> None:
        pass