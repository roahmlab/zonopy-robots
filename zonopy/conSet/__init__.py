"""
Continuous set representation
Author: Yongseok Kwon
"""
import torch 

class __DefaultOptions(object):
    __dtype: type = torch.float
    __itype: type = torch.int
    __device: str = 'cpu'
    def __repr__(self):
        return f'Default Options of Continuous Set\n - dtype: {self.__dtype}\n - itype: {self.__itype}\n - device: {self.__device}'
    def __str__(self):
        return self.__repr__()
    @property
    def DTYPE(self):
        return self.__dtype
    @property
    def ITYPE(self):
        return self.__itype
    @property
    def DEVICE(self):
        return self.__device
    def set(self, dtype=None,itype=None,device=None):
        if dtype is not None:
            if dtype == float:
                dtype = torch.double
            assert dtype == torch.float or dtype == torch.double, 'Default dtype should be float.'
            self.__dtype = dtype
        if itype is not None:
            if itype == int:
                itype = torch.int
            assert itype == torch.int or itype == torch.long or itype == torch.short, 'Default itype should be integer.'
            self.__itype = itype
        if device is not None:
            PROPERTY_ID.to(device)
            self.__device = device

DEFAULT_OPTS = __DefaultOptions()


class __Property_ID(object):
    __dict: dict = {'None':[]}
    __properties: list = []
    __ids: torch.Tensor = torch.tensor([],dtype=torch.long)
    __device: str = 'cpu' 

    def _reset(self,device='cpu'):
        self.__dict = {'None':torch.tensor([],dtype=torch.long,device=device)}
        self.__properties = []
        self.__ids = torch.tensor([],dtype=torch.long,device=device)
        self.__device = device

    def __getitem__(self,key): 
        assert isinstance(key,str)       
        return torch.tensor(self.__dict[key],dtype=torch.long)
    @property 
    def dictionary(self):
        return self.__dict
    @property 
    def properties(self):
        return self.__properties 
    @property 
    def ids(self):
        return self.__ids
    @property 
    def device(self):
        return self.__device    
    @property
    def offset(self):
        return self.__ids.numel()
    def __repr__(self):
        return str(self.__dict)
    def __str__(self):
        return self.__repr__()
    def to(self,device):
        self.__ids = self.__ids.to(device=device)
        for key, item in self.__dict.items():
            self.__dict[key] = item.to(deivce=device)

    def update(self,num,prop='None'):
        if isinstance(prop,str):
            new_id = torch.arange(num,dtype=torch.long) + self.offset
            self.__ids = torch.hstack((self.__ids,new_id))
            if prop in self.__properties:
                self.__dict[prop] = torch.hstack((self.__dict[prop],new_id))
            else:
                self.__properties.append(prop)
                self.__dict[prop] = new_id.clone()
        '''
        elif isinstance(prop,dict):
            # prop = prop names : # gens
            ct, ct_prev = 0, 0
            for pro in prop:
                if pro in self.__properties:
                    ct_prev = ct
                    ct = ct_prev + prop[pro]
                    self.__dict[pro].extend(new_id[ct:ct_prev].tolist())
                else:
                    ct_prev = ct
                    ct = ct_prev + prop[pro]
                    self.__properties.append(pro)
                    self.__dict[pro] = new_id[ct:ct_prev].tolist()
        '''


        return new_id

PROPERTY_ID = __Property_ID()

    




