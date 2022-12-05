from transformers import ViTMAEConfig, ViTMAEModel, ViTMAEForPreTraining
from dataloader import BATCH_SIZE, xrd_dataloader
import torch
from torch import nn, optim
import scipy
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initializing a ViT MAE vit-mae-base style configuration
configuration = ViTMAEConfig(
    image_size=100,
    num_channels=1,
    hidden_size=480,
    intermediate_size=1024,
    decoder_intermediate_size=1024,
    patch_size=10
)

# Initializing a model (with random weights) from the vit-mae-base style configuration
model = ViTMAEForPreTraining(configuration)
configuration = model.config
model= nn.DataParallel(model)
model.to(device)
mse_loss = nn.MSELoss()

# Accessing the model configuration


print('----------------------------------------------------------------')

for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print('----------------------------------------------------------------')
print('number of parameters: ', sum(p.numel() for p in model.parameters()))

def tile(sample):
    edge_sqrt = 10
    tile = np.reshape(np.arange(edge_sqrt ** 2), (edge_sqrt,edge_sqrt))
    a = np.repeat(tile, edge_sqrt, axis=0)
    b = np.repeat(a, edge_sqrt, axis=1)
    tiled = np.tile(tile, (edge_sqrt, edge_sqrt))
    arr = tiled + b * edge_sqrt ** 2
    sample = sample[arr]

    return sample.reshape(1, 100, 100)


def train_model(num_epochs=100):
    outputs = []
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    for epoch in range(num_epochs):
        for idx, data in enumerate(xrd_dataloader):
            gaussian_data = []
            for xrd in data:
                gaussian_data.append(scipy.ndimage.gaussian_filter1d(xrd, 4))
            gaussian_data = np.array(gaussian_data)

            data_tiled = []
            
            for sample in data:
                data_tiled.append(tile(sample[0].numpy()))
            data = np.array(data_tiled)

            gaussian_data_tiled = []
            for sample in gaussian_data:
                gaussian_data_tiled.append(tile(sample[0]))
            
            gaussian_data = np.array(gaussian_data_tiled)
            
            data = torch.from_numpy(data).float()
            gaussian_data = torch.from_numpy(gaussian_data).float()
            # ===================forward=====================
            gaussian_data = gaussian_data.to(device)
            output = model(gaussian_data)
            loss = mse_loss(output.logits, data.squeeze())
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 2500 == 0:
                torch.save(model.state_dict(), f'/pscratch/sd/h/hasitha/xrd/vitmae_xrd_gaussian_blured/vitmae_xrd_gaussian_epoch_{epoch}_batch_{idx}.pth')
            if idx % 5 == 0:
                print(f"Finished batch {idx} in epoch {epoch + 1}. Loss: {loss.item():.4f}")

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))
        outputs.append((epoch, data, output))



# Train the model

model.train(True)
train_model(num_epochs=100)
model.train(False)

print('Finished Training, saving the model')
torch.save(model.state_dict(), '/global/homes/h/hasitha/latent_xrd/vitmae_xrd_gaussian.pth')
print('Model saved')