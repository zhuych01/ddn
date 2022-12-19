import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True
data = torch.randn([256, 3, 224, 224], dtype=torch.float, device='cuda', requires_grad=True)
net = torch.nn.Conv2d(3, 64, kernel_size=[7, 7], padding=[3, 3], stride=[2, 2], dilation=[1, 1], groups=1)
net = net.cuda().float()
out = net(data)
out.backward(torch.randn_like(out))
torch.cuda.synchronize()