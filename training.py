import torch
import numpy as np
import matplotlib.pyplot as plt
from sampling import sample_plot_image
from torch.optim import Adam
from SimpleUnet import SimpleUnet
from noise_scheduler import get_loss, T, BATCH_SIZE, dataloader

model = SimpleUnet()
device = "cuda" #if torch.cuda.is_avaliable() else "cpu"
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 50
PATH = 'model/modelModel50.pt'

#Training
averageArray = []
for epoch in range(epochs):
    total = 0
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        #print("Batch: ", batch.size())
        t = torch.randint(0, T, (BATCH_SIZE,), device=device).long() #t should be the random time step
        #print("Other t: ", t)
        loss = get_loss(model, batch, t)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
        total += loss.item() 
        if step == 14:
            averageArray.append(total/14)
        # if epoch % 50 == 0 and step == 0:
        #     print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
        #     sample_plot_image(model)

x = np.arange(1, epoch+2)
plt.title("Average Loss in each Epoch")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.plot(x, averageArray)
plt.show()

# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())        

torch.save(model.state_dict(), PATH)