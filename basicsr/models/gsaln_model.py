import torch
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel

@MODEL_REGISTRY.register()
class GSALNModel(SRModel):
    def __init__(self, opt):
        super().__init__(opt)
    def test(self):
        tta = self.opt.get('val', {}).get('tta_x8', False)
        if not tta:
            return super().test()
        net = self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g
        training_mode = not hasattr(self, 'net_g_ema')
        net.eval()
        with torch.no_grad():
            x = self.lq
            l = [x, torch.flip(x, dims=[-1]), torch.flip(x, dims=[-2]), torch.flip(x, dims=[-2, -1])]
            xt = x.transpose(-2, -1)
            l += [xt, torch.flip(xt, dims=[-1]), torch.flip(xt, dims=[-2]), torch.flip(xt, dims=[-2, -1])]
            outs = []
            for i, inp in enumerate(l):
                y = net(inp)
                if i >= 4:
                    y = y.transpose(-2, -1)
                    j = i - 4
                else:
                    j = i
                if j == 1:
                    y = torch.flip(y, dims=[-1])
                elif j == 2:
                    y = torch.flip(y, dims=[-2])
                elif j == 3:
                    y = torch.flip(y, dims=[-2, -1])
                outs.append(y)
            self.output = torch.mean(torch.stack(outs, dim=0), dim=0)
        if training_mode:
            net.train()
