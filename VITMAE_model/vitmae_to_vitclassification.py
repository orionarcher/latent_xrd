from transformers import ViTMAEConfig, ViTMAEModel, ViTMAEForPreTraining

import torch
from torch import nn

# Initializing a ViT MAE vit-mae-base style configuration 
configuration = ViTMAEConfig(
    image_size=100,
    num_channels=1,
    hidden_size=480,
    intermediate_size=1024,
    decoder_intermediate_size=1024,
    patch_size=10,
    mask_ratio =0.75 # default mask ration is 0.75
)

# # # Initializing a model (with random weights) from the vit-mae-base style configuration
model = ViTMAEForPreTraining(configuration)
configuration = model.config
model= nn.DataParallel(model)

path = './vitame_xrd.pth'

model.load_state_dict(torch.load(path,map_location='cpu'))
model.module.state_dict()
torch.save(model.module.state_dict(), './vitmae_xrd.pth')

configuration = ViTMAEConfig(
    image_size=100,
    num_channels=1,
    hidden_size=480,
    intermediate_size=1024,
    decoder_intermediate_size=1024,
    patch_size=10,
    mask_ratio =0.75,
    num_labels =7 # adding the number of classes to be classsified to the configuration
)

# Initializing a model (with random weights) from the vit-mae-base style configuration
model = ViTMAEForPreTraining(configuration)
configuration = model.config

model.load_state_dict(torch.load("./vitmae_xrd.pth",map_location='cpu'))

model.save_pretrained("./classification_xrd/")