from torch import nn

BACKBONE_MODULES = ("stem", "context1", "context2", "context3",
                    "context_down1", "context_down2", "fusion1", "fusion2", "fusion3")

class UploadedBackbone(nn.Module):
    def forward(self, x, *, return_pyramid=False):
        if return_pyramid:
            low_level, spatial = self.stem(x, return_low_level=True)
        else:
            spatial = self.stem(x)
        c1 = self.context1(spatial)
        f1 = self.fusion1(spatial=spatial, context=c1)
        c2 = self.context2(self.context_down1(c1))
        f2 = self.fusion2(spatial=f1, context=c2)
        c3 = self.context3(self.context_down2(c2))
        f3 = self.fusion3(spatial=f2, context=c3)
        if return_pyramid:
            return low_level, f2, f3
        return f1, f2, f3, c2
