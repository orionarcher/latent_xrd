# -------------------------importing required packages--------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataloader import xrd_dataloader, binary_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Autoencorders(nn.Module):
    """constructs the Autoencorder.
    
    projects the 10000 dimension XRD data to 512 latent space
    """

    def __init__(self):
        super(Autoencorders, self).__init__()
        # encorder architecture
        self.encoder = nn.Sequential(
            nn.Linear(10000, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True))

        # decorder architecture
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 10000),
            nn.Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


model = Autoencorders()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model= nn.DataParallel(model)
model.to(device)
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

print('----------------------------------------------------------------')

for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print('----------------------------------------------------------------')
print('number of parameters: ', sum(p.numel() for p in model.parameters()))


print('start training')
num_epochs = 100
outputs = []
for epoch in range(num_epochs):
    lst_loss = []
    for data in binary_dataloader:
        data = data.float()
        data = data.to(device)
        # ===================forward=====================
        output = model(data)
        loss = mse_loss(output, data)
        lst_loss.append(loss.item())
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('epoch [{}/{}], loss:{:.7f}'.format(epoch + 1, num_epochs, sum(lst_loss)/len(lst_loss)))
    outputs.append((epoch, data, output))

print('Finished Training, saving the model')
# change the path accordingly
torch.save(model.state_dict(), './vanilla.pth')
print('Model saved')