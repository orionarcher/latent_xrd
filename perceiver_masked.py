import torch
from transformers import (
    PerceiverModel,
    PerceiverConfig,
)
from transformers.models.perceiver.modeling_perceiver import PerceiverBasicDecoder

from torch import nn, optim

from dataloader import xrd_dataloader
from dataloader import XRDDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Set up model

config = PerceiverConfig(d_latent = 128, d_model=10000)
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

mse_loss = nn.MSELoss()

def train_model(num_epochs=100):
    MASK_TOKEN = -0.01
    WINDOW_SIZE = 500
    XRD_SPECTRA_SIZE = 10000

    outputs = []
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    for epoch in range(num_epochs):
        for idx, data in enumerate(xrd_dataloader):
            data = data.float()

            # Apply a mask to the batch
            masked_data = data.clone()
            batch_size = masked_data.size()[0]
            window_start = torch.randint(0, XRD_SPECTRA_SIZE - WINDOW_SIZE, (1,))[0]
            masked_data[:, :, window_start:window_start + WINDOW_SIZE] = torch.full((batch_size, 1, WINDOW_SIZE), MASK_TOKEN)

            # ===================forward=====================
            output = model(inputs=data)
            loss = mse_loss(output.logits, data)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 5 == 0:
                print(f"Finished batch {idx} in epoch {epoch + 1}. Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch + 1}/{num_epochs}], loss:{loss.item():.4f}")
        outputs.append((epoch, data, output))


model.train(True)
train_model(num_epochs=100)
model.train(False)

print('Finished Training, saving the model')
torch.save(model.state_dict(), '/global/homes/h/hasitha/latent_xrd/masked_perceiver.pth')
print('Model saved')

