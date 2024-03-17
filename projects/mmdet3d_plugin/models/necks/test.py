import torch
import torch.nn.functional as F

B, C, H, W = 2, 2, 3, 4
x = flow = torch.arange(48).reshape(B, C, H, W).float()

# mesh grid
xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
grid = torch.cat((xx, yy), dim=1).float()

# update flow with grid
flow += grid

# permute flow dimensions
flow = flow.permute(0, 2, 3, 1)

# normalize flow to [-1, 1]
flow[..., 0] = flow[..., 0] / (W - 1) * 2 - 1.0
flow[..., 1] = flow[..., 1] / (H - 1) * 2 - 1.0
flow[0,0,0,0]=0.5
flow[0,0,0,1]=0.5
print(x)
print(flow)

# apply grid_sample
x = F.grid_sample(x, flow, mode='bilinear', align_corners=True)

print(x)



