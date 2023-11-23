import numpy as np

class rmsprop:
    def __init__(self, w_shape, b_shape, beta=0.9) -> None:
        self.beta = beta
        self.sdw = np.zeros(w_shape)
        self.sdb = np.zeros(b_shape)

    
    def get_dw_opt(self, dw):
        self.sdw = self.beta*self.sdw + (1-self.beta)*dw**2

        # For numerical stability
        epsilon = 1e-8
        return dw/np.sqrt(self.sdw + epsilon)
    

    def get_db_opt(self, db):
        self.sdb = self.beta*self.sdb + (1-self.beta)*db**2

        # For numerical stability
        epsilon = 1e-8
        return db/np.sqrt(self.sdb + epsilon)
    
class sgd:
    def __init__(self) -> None:
        pass
    def get_dw_opt(self, dw):
        return dw
    def get_db_opt(self, db):
        return db

class adam:
    def __init__(self, w_shape, b_shape, beta1=0.9, beta2=0.999) -> None:
        pass


def get_optimizer(opt):
        if opt == 'rmsprop':
            return rmsprop(w_shape=self.w.shape, b_shape=self.b.shape)
        elif opt == 'adam':
            return adam(w_shape=self.w.shape, b_shape=self.b.shape)
        elif opt == 'sgd':
            return sgd()
        else:
            raise Exception('\'' + str(opt) + '\' optimizer not found!')