
from transformers import ViTForImageClassification
# from dataloader import BATCH_SIZE, square_xrd_dataloader_gaussian
import torch

import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = './classification_xrd_final'

model = ViTForImageClassification.from_pretrained(path )

model.load_state_dict(torch.load("./test_model.pth",map_location='cpu'))
model= nn.DataParallel(model)
model.to(device)

from dataloader import test_square_xrd_classification_dataloader
total_sum = 0
total_len = 0
total_len = total_len.to(device)
total_sum = total_sum.to(device)

for idx, data in enumerate(test_square_xrd_classification_dataloader):
        classification = data[:, :, -1, 0]-1
        
        data = data[:, :, :-1, :]
        data = data.to(device)
        classification = classification.to(device)
        output = model(data)
        
        output = np.argmax(output.logits.detach().numpy(), axis=1)
        cache = output == classification.numpy().astype(int).squeeze()
        total_sum += sum(cache)
        total_len += len(cache)
        if idx%20 ==0:
                print(f' batch {idx}, accuracy = {total_sum/total_len}')
print(f'accuracy = {total_sum/total_len}')
