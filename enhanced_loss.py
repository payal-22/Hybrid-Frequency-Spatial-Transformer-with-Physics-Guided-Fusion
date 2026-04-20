
import torch, torch.nn as nn, torch.nn.functional as F

class SSIMLoss(nn.Module):
    def __init__(self, ws=11):
        super().__init__()
        self.ws = ws
    def _window(self, size, device):
        coords = torch.arange(size, dtype=torch.float32, device=device) - size//2
        g = torch.exp(-(coords**2)/(2*1.5**2)); g /= g.sum()
        return g.outer(g).unsqueeze(0).unsqueeze(0)
    def forward(self, p, t):
        C1, C2, pad = 0.01**2, 0.03**2, self.ws//2
        w = self._window(self.ws, p.device).expand(p.size(1),-1,-1,-1)
        mu1 = F.conv2d(p,w,padding=pad,groups=p.size(1))
        mu2 = F.conv2d(t,w,padding=pad,groups=p.size(1))
        s1  = F.conv2d(p*p,w,padding=pad,groups=p.size(1)) - mu1**2
        s2  = F.conv2d(t*t,w,padding=pad,groups=p.size(1)) - mu2**2
        s12 = F.conv2d(p*t,w,padding=pad,groups=p.size(1)) - mu1*mu2
        return 1 - ((2*mu1*mu2+C1)*(2*s12+C2)/((mu1**2+mu2**2+C1)*(s1+s2+C2))).mean()

class TotalUIELoss(nn.Module):
    def __init__(self): super().__init__(); self.ssim = SSIMLoss()
    def forward(self, I_rec, J, T, A, gt, I_input):
        l1   = F.l1_loss(I_rec, gt)
        l2   = F.mse_loss(I_rec, gt)
        ssim = self.ssim(I_rec, gt)
        phy  = F.l1_loss(J*T + A*(1-T), I_input)
        j_l1 = F.l1_loss(J, gt)
        total = 0.8*l1 + 0.2*l2 + 0.3*ssim + 0.1*phy + 0.1*j_l1
        return total, {'l1':l1.item(),'l2':l2.item(),'ssim':ssim.item(),'physics':phy.item()}
