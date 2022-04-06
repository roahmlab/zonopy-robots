"""
Define class for zonotope
Author: Yongseok Kwon
Reference: CORA
"""
import torch
import matplotlib.patches as patches
from zonopy.conSet import DEFAULT_OPTS
from zonopy.conSet.polynomial_zonotope.poly_zono import polyZonotope 
from zonopy.conSet.interval.interval import interval
from zonopy.conSet.utils import delete_column
from zonopy.conSet.zonotope.utils import pickedGenerators, ndimCross

EMPTY_TENSOR = torch.tensor([])
class zonotope:
    '''
    zono: <zonotope>, <torch.float64>

    Z: <torch.Tensor> center vector and generator matrix Z = [c,G]
    , shape [nx, N+1] OR [nx], where N = 0
    -> shape [nx, N+1]
    dtype: data type of class properties
    , torch.float or torch.double
    device: device for torch
    , 'cpu', 'gpu', 'cuda', ...
    center: <torch.Tensor> center vector
    , shape [nx,1] 
    generators: <torch.Tensor> generator matrix
    , shape [nx, N]
        
    
    Eq. (coeff. a1,a2,...,aN \in [0,1])
    G = [g1,g2,...,gN]
    zono = c + a1*g1 + a2*g2 + ... + aN*gN
    '''
    def __init__(self,Z=EMPTY_TENSOR,dtype=None,device=None):
        if dtype is None:
            dtype = DEFAULT_OPTS.DTYPE
        if device is None:
            device = DEFAULT_OPTS.DEVICE
        if isinstance(Z,list):
            Z = torch.tensor(Z)
        assert isinstance(Z,torch.Tensor), f'The input matrix should be either torch tensor or list, but {type(Z)}.'
        if len(Z.shape) == 1:
            Z = Z.reshape(1,-1)
        if dtype == float:
            dtype = torch.double
        assert dtype == torch.float or dtype == torch.double, f'dtype should be either torch.float (torch.float32) or torch.double (torch.float64), but {dtype}.'
        assert len(Z.shape) == 2, f'The dimension of Z input should be either 1 or 2, but {len(Z.shape)}.'

        self.__dtype = dtype
        self.__device = device
        self.Z = Z.to(dtype=dtype,device=device)
        self.__center = self.Z[:,0]
        self.__generators = self.Z[:,1:]
    @property
    def dtype(self):
        return self.__dtype
    @property
    def device(self):
        return self.__device
    @property
    def center(self):
        return self.__center
    @center.setter
    def center(self,value):
        self.Z[:,0] = self.__center = value.to(dtype=self.__dtype,device=self.__device)
    @property
    def generators(self):
        return self.__generators
    @generators.setter
    def generators(self,value):
        value = value.to(dtype=self.__dtype,device=self.__device)
        self.Z = torch.hstack((self.__center.reshape(-1,1),value))
        self.__generators = value
    @property
    def dimension(self):
        return self.Z.shape[0]
    @property
    def n_generators(self):
        return self.__generators.shape[1]

    def to(self,dtype=None,device=None):
        if dtype is None:
            dtype = self.__dtype
        if device is None:
            device = self.__device
        return zonotope(self.Z,dtype,device)
    
    def __str__(self):
        zono_str = f"""center: \n{self.center.to(dtype=torch.float,device='cpu')} \n\nnumber of generators: {self.n_generators} 
            \ngenerators: \n{self.generators.to(dtype=torch.float,device='cpu')} \n\ndimension: {self.dimension}\ndtype: {self.dtype} \ndevice: {self.device}"""
        del_dict = {'tensor':' ','    ':' ','(':'',')':''}
        for del_el in del_dict.keys():
            zono_str = zono_str.replace(del_el,del_dict[del_el])
        return zono_str
    
    def __repr__(self):
        return str(self.Z).replace('tensor','zonotope')
    
    def  __add__(self,other):
        '''
        Overloaded '+' operator for Minkowski sum
        self: <zonotope>
        other: <torch.tensor> OR <zonotope>
        return <polyZonotope>
        '''   
        if isinstance(other, torch.Tensor):
            Z = torch.clone(self.Z)
            assert other.shape == Z[:,0].shape, f'array dimension does not match: should be {Z[:,0].shape}, but {other.shape}.'
            Z[:,0] += other
                
        elif isinstance(other, zonotope): 
            assert self.dimension == other.dimension, f'zonotope dimension does not match: {self.dimension} and {other.dimension}.'
            Z = torch.hstack([(self.center + other.center).reshape(-1,1),self.generators,other.generators])
        else:
            assert False, f'the other object is neither a zonotope nor a torch tensor: {type(other)}.'

        return zonotope(Z,self.__dtype,self.__device)

    __radd__ = __add__
    def __sub__(self,other):
        return self.__add__(-other)
    def __rsub__(self,other):
        return -self.__sub__(other)
    def __iadd__(self,other): 
        return self+other
    def __isub__(self,other):
        return self-other
    def __pos__(self):
        return self    
    
    def __neg__(self):
        '''
        Overloaded unary '-' operator for negation
        self: <zonotope>
        return <zonotope>
        '''   
        Z = -self.Z
        Z[:,1:] = self.Z[:,1:]
        return zonotope(Z,self.__dtype,self.__device)    
    
    def __rmatmul__(self,other):
        '''
        Overloaded '@' operator for matrix multiplication
        self: <zonotope>
        other: <torch.tensor>
        
        zono = other @ self

        return <zonotope>
        '''   
        assert isinstance(other, torch.Tensor), f'The other object should be torch tensor, but {type(other)}.'
        other = other.to(dtype=self.__dtype,device=self.__device)
        Z = other @ self.Z
        return zonotope(Z,self.__dtype,self.__device)

    def __matmul__(self,other):
        assert isinstance(other, torch.Tensor), f'The other object should be torch tensor, but {type(other)}.'
        other = other.to(dtype=self.__dtype,device=self.__device)
        Z = self.Z @ other
        return zonotope(Z,self.__dtype,self.__device)   
        
    def slice(self,slice_dim,slice_pt):
        '''
        slice zonotope on specified point in a certain dimension
        self: <zonotope>
        slice_dim: <torch.Tensor> or <list> or <int>
        , shape  []
        slice_pt: <torch.Tensor> or <list> or <float> or <int>
        , shape  []
        return <zonotope>
        '''
        if isinstance(slice_dim, list):
            slice_dim = torch.tensor(slice_dim,dtype=int,device=self.__device)
        elif isinstance(slice_dim, int) or (isinstance(slice_dim, torch.Tensor) and len(slice_dim.shape)==0):
            slice_dim = torch.tensor([slice_dim],dtype=int,device=self.__device)

        if isinstance(slice_pt, list):
            slice_pt = torch.tensor(slice_pt,dtype=self.__dtype,device=self.__device)
        elif isinstance(slice_pt, int) or isinstance(slice_pt, float) or (isinstance(slice_pt, torch.Tensor) and len(slice_pt.shape)==0):
            slice_pt = torch.tensor([slice_pt],dtype=self.__dtype,device=self.__device)

        assert isinstance(slice_dim, torch.Tensor) and isinstance(slice_pt, torch.Tensor), 'Invalid type of input'
        assert len(slice_dim.shape) ==1, 'slicing dimension should be 1-dim component.'
        assert len(slice_pt.shape) ==1, 'slicing point should be 1-dim component.'
        assert len(slice_dim) == len(slice_pt), f'The number of slicing dimension ({len(slice_dim)}) and the number of slicing point ({len(slice_dim)}) should be the same.'

        N = len(slice_dim)
        
        Z = self.Z
        c = self.center
        G = self.generators

        slice_idx = torch.zeros(N,dtype=int,device=self.__device)
        for i in range(N):
            non_zero_idx = (G[slice_dim[i],:] != 0).nonzero().reshape(-1)
            if len(non_zero_idx) != 1:
                if len(non_zero_idx) == 0:
                    raise ValueError('no generator for slice index')
                else:
                    raise ValueError('more than one generators for slice index')
            slice_idx[i] = non_zero_idx

        slice_c = c[slice_dim]

        slice_G = torch.zeros(N,N,dtype=self.__dtype,device=self.__device)
        for i in range(N):
            slice_G[i] = G[slice_dim[i],slice_idx]
        
        slice_lambda = torch.linalg.solve(slice_G, slice_pt - slice_c)

        assert not any(abs(slice_lambda)>1), 'slice point is ouside bounds of reach set, and therefore is not verified'

        Z_new = torch.zeros(Z.shape,dtype=self.__dtype,device=self.__device)
        Z_new[:,0] = c + G[:,slice_idx]@slice_lambda
        Z_new[:,1:] = G
        Z_new = delete_column(Z_new,slice_idx+1)
        return zonotope(Z_new,self.__dtype,self.__device)

    def project(self,dim=[0,1]):
        '''
        the projection of a zonotope onto the specified dimensions
        self: <zonotope>
        dim: <int> or <list> or <torch.Tensor> dimensions for prjection 
        
        return <zonotope>
        '''
        Z = self.Z[dim,:]
        return zonotope(Z,self.__dtype,self.__device)

    def polygon(self):
        '''
        converts a 2-d zonotope into a polygon as vertices
        self: <zonotope>

        return <torch.Tensor>, <torch.float64>
        '''
        dim = 2
        z = self.deleteZerosGenerators()
        c = z.center
        G = torch.clone(z.generators)
        n = z.n_generators
        x_max = torch.sum(abs(G[0,:]))
        y_max = torch.sum(abs(G[1,:]))
        G[:,z.generators[1,:]<0] = - z.generators[:,z.generators[1,:]<0] # make all y components as positive
        angles = torch.atan2(G[1,:], G[0,:])
        ang_idx = torch.argsort(angles)
        
        vertices_half = torch.zeros(dim,n+1,dtype=self.__dtype,device=self.__device)
        for i in range(n):
            vertices_half[:,i+1] = vertices_half[:,i] + 2*G[:,ang_idx[i]]
        
        vertices_half[0,:] += (x_max-torch.max(vertices_half[0,:]))
        vertices_half[1,:] -= y_max

        full_vertices = torch.zeros(dim,2*n+1,dtype=self.__dtype,device=self.__device)
        full_vertices[:,:n+1] = vertices_half
        full_vertices[:,n+1:] = -vertices_half[:,1:] + vertices_half[:,0].reshape(dim,1) + vertices_half[:,-1].reshape(dim,1) #flipped
        
        full_vertices += c.reshape(dim,1)
        return full_vertices.to(dtype=self.__dtype,device=self.__device)
    def polytope(self):
        '''
        converts a zonotope from a G- to a H- representation
        P
        comb
        isDeg
        '''

        #z = self.deleteZerosGenerators()
        c = self.center
        G = torch.clone(self.generators)
        h = torch.linalg.vector_norm(G,dim=0)
        h_sort, indicies = torch.sort(h,descending=True)
        h_zero = h_sort < 1e-6
        if torch.any(h_zero):
            first_reduce_idx = torch.nonzero(h_zero)[0,0]
            Gunred = G[:,indicies[:first_reduce_idx]]
            # Gred = G[:,indicies[first_reduce_idx:]]
            # d = torch.sum(abs(Gred),1)
            # G = torch.hstack((Gunred,torch.diag(d)))
            G = Gunred

        dim, n_gens = G.shape

        '''
        if maxcombs is None:
            comb = co
        
        elif n_gens > maxcombs:
        '''
        
        
                
                
        if dim == 1:
            C = (G/torch.linalg.vector_norm(G,dim=0)).T
        elif dim == 2:      
            C = torch.vstack((-G[1,:],G[0,:]))
            C = (C/torch.linalg.vector_norm(C,dim=0)).T
        elif dim == 3:
            # not complete for example when n_gens < dim-1; n_gens =0 or n_gens =1 
            comb = torch.combinations(torch.arange(n_gens,device=self.__device),r=dim-1)
            Q = torch.vstack((G[:,comb[:,0]],G[:,comb[:,1]]))
            C = torch.vstack((Q[1,:]*Q[5,:] - Q[2,:]*Q[4,:],-Q[0,:]*Q[5,:] - Q[2,:]*Q[3,:],Q[0,:]*Q[4,:] - Q[1,:]*Q[3,:]))
            C = (C/torch.linalg.vector_norm(C,dim=0)).T
        elif dim >=4 and dim<=7:
            assert False
        else:
            assert False
        
        index = torch.sum(torch.isnan(C),dim=1) == 0
        C = C[index]
        deltaD = torch.sum(abs(C@G),dim=1)
        d = (C@c)
        PA = torch.vstack((C,-C))
        Pb = torch.hstack((d+deltaD,-d+deltaD))

        return PA, Pb, C
        '''
        dim, n_gens = G.shape
        if torch.matrix_rank(G) >= dim:
            if dim > 1:
                comb = torch.combinations(torch.arange(n_gens,device=self.__device),r=dim-1)
                n_comb = len(comb)
                C = torch.zeros(n_comb,dim, device=self.__device)
                for i in range(n_comb):
                    indices = comb[i,:]
                    Q = G[:,indices]
                    v = ndimCross(Q)
                    C[i,:] = v/torch.linalg.norm(v)
                # remove None rows dues to rank deficiency
                index = torch.sum(torch.isnan(C),axis=1) == 0
                C = C[index,:]
            else: 
                C =torch.eye(1,device=self.__device)

            # build d vector and determine delta d
            deltaD = torch.zeros(len(C),device=self.__device)
            for iGen in range(n_gens):
                deltaD += abs(C@G[:,iGen])
            # compute dPos, dNeg
            dPos, dNeg = C@c + deltaD, - C@c + deltaD
            # construct the overall inequality constraints
            C = torch.hstack((C,-C))
            d = torch.hstack((dPos,dNeg))
            # catch the case where the zonotope is not full-dimensional
            temp = torch.min(torch.sum(abs(C-C[0]),1),torch.sum(abs(C+C[0]),1))
            if dim > 1 and (C.numel() == 0 or torch.all(temp<1e-12) or torch.all(torch.isnan(C)) or torch.any(torch.max(abs(C),0).values<1e-12)):
                S,V,_ = torch.linalg.svd(G)

                Z_ = S.T@torch.hstack((c,G))

                ind = V <= 1e-12

                # 1:len(V) 

                
        return P, comb, isDeg
        '''
    def deleteZerosGenerators(self,eps=0):
        '''
        delete zero vector generators
        self: <zonotope>

        return <zonotope>
        '''
        non_zero_idxs = torch.any(abs(self.generators)>eps,axis=0)
        Z_new = torch.hstack((self.center.reshape(-1,1),self.generators[:,non_zero_idxs]))
        return zonotope(Z_new,dtype=self.__dtype,device=self.__device)

    def plot(self, ax,facecolor='none',edgecolor='green',linewidth=.2,dim=[0,1]):
        '''
        plot 2 dimensional projection of a zonotope
        self: <zonotope>
        ax: <Axes> axes oject of a figure to plot
        facecolor: <string> color of face
        edgecolor: <string> color of edges

        ex.
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.gca()
        zono.plot(ax)
        plt.show()
        '''
        
        z = self.project(dim)
        p = z.polygon()

        ax.add_patch(patches.Polygon(p.T,alpha=.5,edgecolor=edgecolor,facecolor=facecolor,linewidth=linewidth))

    def to_polyZonotope(self,dim=None,prop='None'):
        '''
        convert zonotope to polynomial zonotope
        self: <zonotope>
        dim: <int>, dimension to take as sliceable
        return <polyZonotope>
        '''
        if dim is None:
            return polyZonotope(self.center,Grest = self.generators,dtype=self.__dtype,device=self.__device)
        assert isinstance(dim,int)
        assert dim <= self.dimension

        g_row_dim =self.generators[dim,:]
        idx = (g_row_dim!=0).nonzero().reshape(-1)
        
        assert idx.numel() != 0, 'no sliceable generator for the dimension.'
        assert idx.numel() == 1,'more than one no sliceable generators for the dimesion.'        
        
        c = self.center
        G = self.generators[:,idx]
        Grest = delete_column(self.generators,idx)

        return polyZonotope(c,G,Grest,dtype=self.__dtype,device=self.__device,prop=prop)
    def to_interval(self):
        c = self.__center
        delta = torch.sum(abs(self.Z),dim=1) - abs(c)
        leftLimit, rightLimit = c -delta, c + delta
        return interval(leftLimit,rightLimit,self.__dtype,self.__device)

    def reduce(self,order,option='girard'):
        if option == 'girard':
            center, Gunred, Gred = pickedGenerators(self,order)
            d = torch.sum(abs(Gred),1)
            Gbox = torch.diag(d)
            ZRed = torch.hstack((center.reshape(-1,1),Gunred,Gbox))
            return zonotope(ZRed,self.dtype,self.device)
        else:
            assert False, 'Invalid reduction option'

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca()
    z = zonotope([[0,1,0],[0,0,1]])
    z.plot(ax)
    plt.show()

    Z = torch.tensor([[0, 1, 0,1],[0, 0, -1,1],[0,0,0,1]])
    z = zonotope(Z)
    print(torch.eye(3)@z)
    print(z-torch.tensor([1,2,3]))
    print(z.Z)
    print(z.slice(2,1).Z)
    print(z)

    #fig = plt.figure()    
   #ax = fig.gca() 
    #z.plot2d(ax)

  
    Z1 = torch.tensor([[0, 1, 0,1,3,4,5,6,7,1,4,4,15,6,1,3],[0, 0, -1,14,5,1,6,7,1,4,33,15,1,2,33,3]])*0.0001
    z1 = zonotope(Z1)
    #z1.plot2d(ax)
    #plt.axis([-5,5,-5,5])
    #plt.show()

