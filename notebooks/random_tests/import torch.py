import torch
from torchmetrics import Accuracy
target1 = torch.tensor([[0, 0, 0, 0],[0, 0, 0, 0]])
target2 = torch.tensor([[0, 1, 2, 3],[0, 1, 2, 3]])
target = torch.cat((target1,target2))
preds1 = torch.tensor([1, 2, 1, 3])
preds2 = torch.tensor([0, 2, 1, 3])
preds = torch.stack((preds1,preds2))
accuracy = Accuracy(mdmc_average='global')
print(accuracy(preds, target))