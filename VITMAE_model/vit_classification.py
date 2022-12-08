# -------------------------importing required packages--------------------------
from transformers import ViTForImageClassification
from dataloader import square_xrd_classification_dataloader, test_square_xrd_classification_dataloader
import torch
import numpy as np
from torch import nn,optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ./classification_xrd/ contains the trained ViTMAE model - run the vitmae_to_vitclassification.py to generate the pretrained vitmae model to be 
model = ViTForImageClassification.from_pretrained("./classification_xrd/")
model= nn.DataParallel(model)
model.to(device)



def train_model(num_epochs=100):
    cross_entropy = nn.CrossEntropyLoss()
    outputs = []
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    for epoch in range(num_epochs):
        lst_loss = []
        for idx, data in enumerate(square_xrd_classification_dataloader):
            classification = data[:, :, -1, 0]-1
            classification_gpu = torch.from_numpy(classification.long().numpy()[:,0])
            data = data[:, :, :-1, :]  
            # ===================forward===================== 
            data = data.to(device)
            classification_gpu = classification_gpu.to(device)
            output = model(data)
            loss = cross_entropy(output.logits, classification_gpu)
            lst_loss.append(loss.item())
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 100 == 0:
                print(f"Finished batch {idx} in epoch {epoch + 1}. Loss: {loss.item():.4f}")
        # test loop
        for idx, data in enumerate(test_square_xrd_classification_dataloader):
            lst_loss_train = []
            classification = data[:, :, -1, 0]-1
            classification_gpu = torch.from_numpy(classification.long().numpy()[:,0])
            data = data[:, :, :-1, :]
            data = data.to(device)
            classification_gpu = classification_gpu.to(device)
            output = model(data)
            loss_train = cross_entropy(output.logits, classification_gpu)
            lst_loss_train.append(loss_train.item())
            
        # printing aggregate loss of training and testing 
        print('epoch [{}/{}], loss:{:.7f}'.format(epoch + 1, num_epochs, sum(lst_loss)/len(lst_loss)))
        print('test time epoch [{}/{}], loss:{:.7f}'.format(epoch + 1, num_epochs, sum(lst_loss_train)/len(lst_loss_train)))
        outputs.append((epoch, data, output))


model.train(True)
train_model(num_epochs=100)
model.train(False)

print('Finished Training, saving the model')
# change the path accordingly
torch.save(model.state_dict(), './vitmae_classification.pth')
print('Model saved')







