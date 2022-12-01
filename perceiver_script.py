import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import (
    PerceiverModel,
    PerceiverConfig,
)
from transformers.models.perceiver.modeling_perceiver import PerceiverBasicDecoder

from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP

from dataloader import xrd_dataloader
from dataloader import XRDDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

######################


# Set up model

config = PerceiverConfig(d_latents = 128, d_model=10000)
decoder = PerceiverBasicDecoder(
    config,
    num_channels=config.d_latents,
    output_num_channels=10000,
    final_project=True,
    trainable_position_encoding_kwargs={
        "num_channels": config.d_latents,
        "index_dims": 1,
    },
)

model = PerceiverModel(config, decoder=decoder)
model= nn.DataParallel(model)
model.to(device)

print('----------------------------------------------------------------')

for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print('----------------------------------------------------------------')
print('number of parameters: ', sum(p.numel() for p in model.parameters()))


mse_loss = nn.MSELoss()

def train_model(num_epochs=100):
    outputs = []
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    for epoch in range(num_epochs):
        for data in xrd_dataloader:
            data = data.float()
            data = data.to(device)
            # ===================forward=====================
            output = model(inputs=data)
            loss = mse_loss(output.logits, data)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))
        outputs.append((epoch, data, output))



# Train the model

model.train(True)
train_model(num_epochs=100)
model.train(False)

print('Finished Training, saving the model')
torch.save(model.state_dict(), '/global/homes/h/hasitha/latent_xrd/perceiver.pth')
print('Model saved')