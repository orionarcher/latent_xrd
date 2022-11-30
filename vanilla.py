import numpy as np
import h5py as h5
import os
import torch
import tqdm

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from dataloader import XRDDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Autoencorders(nn.Module):
    def __init__(self):
        super(Autoencorders, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10005, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True))


        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 10005),
            nn.Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


model = Autoencorders()
model= nn.DataParallel(model)
model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

print('----------------------------------------------------------------')

for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print('----------------------------------------------------------------')
print('number of parameters: ', sum(p.numel() for p in model.parameters()))

XRD_DATA_PATH = '/pscratch/sd/h/hasitha/xrd/icsd_data_189476_10000_cry_extinction_space_density_vol.h5' # path to the data file
xrd_dataset = XRDDataset(XRD_DATA_PATH)
xrd_dataloader = DataLoader(
    xrd_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)


print('start training')
num_epochs = 1
outputs = []
for epoch in range(num_epochs):
    for data in xrd_dataloader:
        data = data.float()
        data = data.to(device)
        # ===================forward=====================
        output = model(data)
        loss = criterion(output, data)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))
    outputs.append((epoch, data, output))

print('Finished Training, saving the model')
torch.save(model.state_dict(), '/global/homes/h/hasitha/latent_xrd/vanilla.pth')
print('Model saved')