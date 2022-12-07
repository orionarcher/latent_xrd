from transformers import ViTForImageClassification
from dataloader import square_xrd_classification_dataloader, test_square_xrd_classification_dataloader
import torch
import numpy as np
from torch import nn,optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTForImageClassification.from_pretrained("./classification_xrd/")
model= nn.DataParallel(model)
model.to(device)



def train_model(num_epochs=100):
    cross_entropy = nn.CrossEntropyLoss()
    # soft_max = nn.Softmax(dim = 1)
    outputs = []
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    for epoch in range(num_epochs):
        lst_loss = []
        for idx, data in enumerate(square_xrd_classification_dataloader):
            classification = data[:, :, -1, 0]-1
            classification_gpu = torch.from_numpy(classification.long().numpy()[:,0])
            data = data[:, :, :-1, :]
            # batch_size = data.shape[0]
            # rows = np.arange(batch_size)
            # one_hot = np.zeros((batch_size, 7))
            # one_hot[rows, classification.int().numpy()[:,0]] = 1  
            # one_hot = torch.from_numpy(one_hot)  
            # ===================forward===================== 
            data = data.to(device)
            classification_gpu = classification_gpu.to(device)
            output = model(data)
            # soft_max_output = soft_max(output.logits)
            loss = cross_entropy(output.logits, classification_gpu)
            lst_loss.append(loss.item())
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 1000 == 0:
                torch.save(model.state_dict(), f'/pscratch/sd/h/hasitha/xrd/vitmae_classification/10_masking_80_data/vitmae_classification_10_mask_80_data_epoch_{epoch}_batch_{idx}.pth')
            if idx % 10 == 0:
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
            

        print('epoch [{}/{}], loss:{:.7f}'.format(epoch + 1, num_epochs, sum(lst_loss)/len(lst_loss)))
        print('train epoch_t [{}/{}], loss:{:.7f}'.format(epoch + 1, num_epochs, sum(lst_loss_train)/len(lst_loss_train)))
        outputs.append((epoch, data, output))


model.train(True)
train_model(num_epochs=100)
model.train(False)

print('Finished Training, saving the model')
torch.save(model.state_dict(), '/global/homes/h/hasitha/latent_xrd/vitmae_classification_10_mask_80_data.pth')
print('Model saved')







