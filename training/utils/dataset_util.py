class ConfigToAttr:  
    def __init__(self, config):  
        for k, v in config.items():  
            setattr(self, k, v)  
        # 也保留原始字典  
        self.opt = config 

    def get(self,key,default):
        return self.opt.get(key,default)