import torch
from SimpleUnet import SimpleUnet
from sampling import sample_plot_image

PATH = "model/modelModel.pt"
model = SimpleUnet()
model.load_state_dict(torch.load(PATH, weights_only=False))
model.eval()
model.to("cuda")

sample_plot_image(model)