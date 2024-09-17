import torch
from sampling import sample_plot_image
from torch.optim import Adam
from SimpleUnet import SimpleUnet
from noise_scheduler import get_loss, T, BATCH_SIZE, dataloader

model = SimpleUnet()
device = "cuda" #if torch.cuda.is_avaliable() else "cpu"
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 151
PATH = 'model/modelModel.pt'

#Training
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        #print("Batch: ", batch.size())
        t = torch.randint(0, T, (BATCH_SIZE,), device=device).long() #t should be the random time step
        #print("Other t: ", t)
        loss = get_loss(model, batch, t)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")

        if epoch % 50 == 0 and step == 0:
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
            sample_plot_image(model)

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())        

torch.save(model.state_dict(), PATH)