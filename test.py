import torch
import gc

from torch.utils.cpp_extension import load
from pathlib import Path

files=[str(f) for f in Path("src").glob("*.cpp")] + [str(f) for f in Path("src").glob("*.cu")]+[str(f) for f in Path("src/culbfgsb").glob("*.cpp")] + [str(f) for f in Path("src/culbfgsb").glob("*.cu")]
test = load(
    'test', files, verbose=False,extra_include_paths=["src","src/culbfgsb"],extra_cflags=["-fpermissive", "-O3"])


class testfunction(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('_lim',torch.tensor(((0.,1.),(0.,2.))))
       
    def forward(self,x):
        return torch.stack((
            x[...,0]*torch.exp(-.1*x[...,1]),
            x[...,0]*torch.exp(-.2*x[...,1]),
            x[...,0]*torch.exp(-.3*x[...,1]),
            x[...,0]*torch.exp(-.5*x[...,1]),
            x[...,0]*torch.exp(-1*x[...,1]),
             x[...,0]*torch.exp(-1.5*x[...,1]),
            x[...,0]*torch.exp(-2*x[...,1]),
        ),-1)
    @property
    def limits(self):
        return self._lim    
for i in range(1,24,4):
    N=int(2**i)
    print(i,N)
    fun=testfunction().double().cuda()
    xtrue= torch.rand(N,2).cuda().double()*0.98+0.01
    xreg = torch.clone(xtrue).double()
    ym=fun(xtrue)
    ym+=0.01*torch.randn_like(ym)
    xl,xu= (i.contiguous().cuda() for i in fun.limits.expand(N,2,2).moveaxis(-1,0))
    nbd=2*torch.ones(xu.numel(),dtype=torch.int32).cuda()


    def objective(x,mask=None):
        a=torch.nn.functional.mse_loss(fun(x),ym,reduction='none').sum(-1)
        b= torch.nn.functional.mse_loss(x,xreg,reduction='none').sum(-1)
        return (a + b).mean()

    def callback(x,summary):
        torch.cuda.synchronize()
        x.grad.fill_(0)
        ret=objective(x)
        ret.backward()
        ret=ret.item()
        #print(ret,torch.any(torch.isnan(x)),torch.any(torch.isnan(x)),x.grad[0],torch.max(x),torch.min(x))
        torch.cuda.synchronize()
        print(f"{summary.num_iteration=} {summary.residual_f=} {summary.residual_g=} {summary.residual_x=} {summary.info=} {ret=}")
        print("x:",torch.any(torch.isnan(x)),torch.min(x),torch.max(x))
        print("grad:",torch.any(torch.isnan(x.grad)),torch.min(x.grad),torch.max(x.grad))



        return ret

    x0=torch.rand_like(xtrue)
    x0.grad = None
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    res=test.run(x0,xl,xu,nbd,callback,20)
    torch.cuda.synchronize()
    gc.collect()
    print()
    print(res,objective(x0)/N)
    print()
    print()


    
    