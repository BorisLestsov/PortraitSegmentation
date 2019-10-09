import torch
from models_seg import ACFDPN
from collections import OrderedDict

example_forward_input = torch.rand(1, 3, 800, 600)

model = ACFDPN(2, backbone="dpn92")
checkpoint = torch.load("model_best.pth.tar")

checkpoint2 = OrderedDict()

for k, v in checkpoint["state_dict"].items():
    name = k
    if k.startswith("module."):
        name = k[7:]
    checkpoint2[name] = v

model.load_state_dict(checkpoint2)

module = torch.jit.trace(model, example_forward_input)
torch.jit.save(module, 'scriptmodule.pt')

m_check = torch.jit.load('scriptmodule.pt')
m_check(example_forward_input)
print("DONE")
