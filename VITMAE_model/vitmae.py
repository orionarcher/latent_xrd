# -------------------------importing required packages--------------------------
from transformers import ViTMAEConfig, ViTMAEModel, ViTMAEForPreTraining
from dataloader import BATCH_SIZE, square_xrd_dataloader
import torch
from torch import nn, optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initializing a ViT MAE vit-mae-base style configuration 
configuration = ViTMAEConfig(
    image_size=100,
    num_channels=1,
    hidden_size=480,
    intermediate_size=1024,
    decoder_intermediate_size=1024,
    patch_size=10,
    mask_ratio=0.75 #default mask ratio is 0.75
)

# Initializing a model (with random weights) from the vit-mae-base style configuration
model = ViTMAEForPreTraining(configuration)
configuration = model.config  # Accessing the model configuration
model= nn.DataParallel(model)  # model data parallelization
model.to(device) 
mse_loss = nn.MSELoss()




print('----------------------------------------------------------------')

for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print('----------------------------------------------------------------')
print('number of parameters: ', sum(p.numel() for p in model.parameters()))

def train_model(num_epochs=100):
    outputs = []
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    for epoch in range(num_epochs):
        lst_loss = []
        for idx, data in enumerate(square_xrd_dataloader):
            # ===================forward=====================
            data = data.to(device)
            output = model(data)
            loss = mse_loss(output.logits, data.squeeze())
            lst_loss.append(loss.item())
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 100 == 0:
                print(f"Finished batch {idx} in epoch {epoch + 1}. Loss: {loss.item():.4f}")

        print('epoch [{}/{}], loss:{:.7f}'.format(epoch + 1, num_epochs, sum(lst_loss)/len(lst_loss)))
        outputs.append((epoch, data, output))



# Train the model
model.train(True)
train_model(num_epochs=100)
model.train(False)

print('Finished Training, saving the model')
#change the save path accordingly
torch.save(model.state_dict(), './vitmae_xrd.pth')
print('Model saved')