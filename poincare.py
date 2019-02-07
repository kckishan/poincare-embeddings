import torch

class PoincareDistance(torch.autograd.Function):
    def __init__(self):
        """
        Function returning distance between points in hyperbolic space.
        
        """
        super(PoincareDistance, self).__init__()
    
    def square_norm(self,x):
        """
        Helper function returning square of the euclidean norm.
        
        Also here we clamp it since it really likes to die to zero.
        
        """
        norm = torch.norm(x,dim=-1,p=2)**2
        return torch.clamp(norm,min=1e-5)
    
    def arcosh(self,x):
        """
        arcosh(x) = log( x + sqrt[x^2 - 1] )
        
        """
        return torch.log(x+torch.sqrt(x*x - 1))

    def forward(self, x, y):
        """
        d(x,y) = arcosh( 1 + 2*z ), where 
        
        z = [ ||x-y||^2 ]/[(1 - ||x||^2)(1 - ||y||^2)]
        
        See equation [1] in the paper.
        
        """
    
        self.save_for_backward(x, y)
        
        z = self.square_norm(x-y)/\
            ((1-self.square_norm(x)) * (1-self.square_norm(y)))
        
        return self.arcosh(1 + 2*z)
                
    def backward(self, g):
        
        u, v = self.saved_tensors
        g = g.unsqueeze(-1)
        
        gu = self.grad(u, v)
        gv = self.grad(v, u)
        
        return g.expand_as(gu) * gu, g.expand_as(gv) * gv

    
        
    def grad(self, theta, x):
        """
        Partial derivative of Poincare distance, see equation [4] in the paper.
        
        d_d(theta,x)/d_theta = A*(B-C), where
        
        A = 4/(beta*sqrt(gamma^2-1))
        
        B = (||x||^2 - 2*<theta,x> + 1)/alpha^2*theta 
        
        C = x/alpha
        
        
        alpha = 1 - ||theta||^2
        
        beta = 1 - ||x||^2
        
        gamma = 1 + 2 * || theta - x ||^2 / (alpha * beta), see equation [3].
        
        
        """
        alpha = (1 - self.square_norm(theta))
        beta = (1 - self.square_norm(x))
        gamma = 1 + 2 * self.square_norm(theta-x) / (alpha * beta)
        
        #very dangerous to compute since we take square of square of very small number!
        A = 4/(beta*torch.sqrt(gamma**2 - 1))
        
        B = ((self.square_norm(x) - 2*torch.sum(theta * x, dim=-1) + 1)/alpha**2).unsqueeze(-1).expand_as(theta)*theta
        
        C = x/alpha.unsqueeze(-1).expand_as(x)
        
        return A.unsqueeze(-1).expand_as(x)*(B-C)


class PoincareEmbedding(torch.nn.Module):
    
    def __init__(self,num_embeddings,embedding_dim=2,eps=1e-5):
        """
        Model class, which stores the embedding, passes inputs to PoincareDistance function
        
        and stores the loss fucntion.
        
        """
        super(PoincareEmbedding, self).__init__()
        
        self.eps = eps #we define the boundary to be 1-eps
        
        self.embedding = torch.nn.Embedding(num_embeddings, 
                                            embedding_dim, 
                                            padding_idx=None, 
                                            max_norm=1-self.eps, 
                                            norm_type=2.0, 
                                            scale_grad_by_freq=False, 
                                            sparse=False)
        
        self.log = []

    def initialize_embedding(self,initial_radius = .001):
        """
        Initialize embedding to be a uniform disk of specified radius.
        
        The algorithm has prooven to be quite sensitive to initial state,
        
        so it would be usefull to keep it here.
        
        """
        
        distibution = torch.distributions.Uniform(-1,1)
        
        x = distibution.sample(self.embedding.weight.data.shape)
        
        self.embedding.weight.data = initial_radius*x/torch.norm(x,p=2,dim=-1).unsqueeze(-1)
        
        
    def forward(self,x,y):
        """
        Looks up the embedding given indexes and compute the distance
        """
        x_emb = self.embedding(x)
        y_emb = self.embedding(y)
        
        dist = PoincareDistance()(x_emb,y_emb)
        
        return dist
    
    
    def loss(self,preds):
        """
        Somewhat elegant solution for the loss fucntion from the original 
        Facebook implementation with an ugly crutch for cuda.
        
        We put positive sample at the position 0, so loss will be minimal for vector:
        
        [1,0,0,....]
        
        At the same time, when we look at the somewhat good negative distance vector:
        
        [-0.001,-10.1,-15,6,....]
        
        It will minimize the loss...
        
        Yes, I am also very surprised it works.
        
        """
        if preds.is_cuda:
            targets = torch.cuda.LongTensor([0]*preds.shape[0])
        else:
            targets = torch.LongTensor([0]*preds.shape[0])
        
        return torch.nn.CrossEntropyLoss(weight=None, 
                          ignore_index=-100, 
                          reduction='mean')(-preds,targets)   

class RiemannianSGD(torch.optim.Optimizer):
    """
    Mostly copied from original implementation.
    
    Riemannian stochastic gradient descent.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
    """

    def __init__(self, params):
        defaults={}
        super(RiemannianSGD, self).__init__(params,defaults)

    def poincare_grad(self,p, d_p):
        """

        Mostly copied from original implementation.

        See equation [5] in the paper.

        Function to compute Riemannian gradient from the
        Euclidean gradient in the Poincaré ball.
        Args:
            p (Tensor): Current point in the ball
            d_p (Tensor): Euclidean gradient at p
        """
    
        p_sqnorm = torch.sum(p.data ** 2, dim=-1, keepdim=True)
        d_p = d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(d_p)
    
        return d_p

    def euclidean_retraction(self,p, d_p, lr):
        p.data.add_(-lr, d_p)  #???  
        return p
        
    def step(self, lr):
        """
        Mostly copied from original implementation.
        
        Performs a single optimization step.
        Arguments:
            lr: learning rate for the current update.
        """
        loss = None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                d_p = p.grad.data #gradient we computed on baclward pass 
                    
                d_p = self.poincare_grad(p, d_p)
                p = self.euclidean_retraction(p, d_p, lr)

        return loss
    