import torch

from perceiver.model.core import (
    PerceiverDecoder,
    PerceiverEncoder,
    PerceiverIO
)

from perceiver.model.core.classifier import ClassificationOutputAdapter
from perceiver.model.core.adapter import TrainableQueryProvider

from perceiver.model.vision.image_classifier import ImageInputAdapter

from torch import nn, optim

from dataloader import xrd_dataloader, binary_dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from perceiver.model.core import (
    PerceiverDecoder,
    PerceiverEncoder,
    PerceiverIO
)

from perceiver.model.core.classifier import ClassificationOutputAdapter
from perceiver.model.core.adapter import TrainableQueryProvider

from perceiver.model.vision.image_classifier import ImageInputAdapter

D_INPUT = 100

# Fourier-encodes pixel positions and flatten along spatial dimensions
input_adapter = ImageInputAdapter(
  image_shape=(D_INPUT, 1),  # M = 224 * 224
  num_frequency_bands=32,
)

# Projects generic Perceiver decoder output to specified number of classes
output_adapter = ClassificationOutputAdapter(
  num_classes=D_INPUT,
  num_output_query_channels=512,  # F
)

# Generic Perceiver encoder
encoder = PerceiverEncoder(
  input_adapter=input_adapter,
  num_latents=512,  # N
  num_latent_channels=512,  # D changed from 1028
  num_cross_attention_qk_channels=input_adapter.num_input_channels,  # C
  num_cross_attention_heads=1,
  num_self_attention_heads=8,
  num_self_attention_layers_per_block=6,
  num_self_attention_blocks=8,
  dropout=0.0,
)

query_provider = TrainableQueryProvider(1, 512) # very arbitrary!

# Generic Perceiver decoder
decoder = PerceiverDecoder(
  output_adapter=output_adapter,
  output_query_provider=query_provider,
  num_latent_channels=512,  # D
  num_cross_attention_heads=1,
  dropout=0.0,
)

# Perceiver IO image classifier
mse_loss = nn.MSELoss()
model = PerceiverIO(encoder, decoder)
model= nn.DataParallel(model)
model.to(device)

print('----------------------------------------------------------------')

for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print('----------------------------------------------------------------')
print('number of parameters: ', sum(p.numel() for p in model.parameters()))


def train_model(num_epochs=100):
    outputs = []
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    for epoch in range(num_epochs):
        for idx, data in enumerate(binary_dataloader):
            batch_size = 32
            data = data.reshape(-1, D_INPUT, 1)
            data = data.float()
            data = data.to(device)
            # ===================forward=====================
            output = model(data)
            loss = mse_loss(output, data.squeeze())
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n_correct = torch.sum(torch.round(output) == data[:, :, 0]) / batch_size

            if idx % 2500 == 0:
                torch.save(model.state_dict(), f'/pscratch/sd/h/hasitha/xrd/perciever_small_random/perceiver_small_random_epoch_{epoch}batch_{idx}.pth')
            if idx % 5 == 0:
                print(f"Finished batch {idx} in epoch {epoch + 1}. Loss: {loss.item():.4f}")
                print(f"The model classified {n_correct:.4f} percent of points correctly.")

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))
        outputs.append((epoch, data, output))



# Train the model

model.train(True)
train_model(num_epochs=100)
model.train(False)

print('Finished Training, saving the model')
torch.save(model.state_dict(), '/global/homes/h/hasitha/latent_xrd/perciever_small_random/perceiver_small_random.pth')
print('Model saved')