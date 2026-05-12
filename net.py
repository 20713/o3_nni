from torch import nn


class Net(nn.Module):
    def __init__(self,in_dim,out_dim=61,hid1=256,hid2=128):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(in_dim,hid1),
            nn.Tanh(),
            nn.Linear(hid1,hid2),
            nn.Tanh(),
            nn.Linear(hid2,out_dim)
        )

    def forward(self,x):
        return self.net(x)
