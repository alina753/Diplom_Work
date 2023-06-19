import torch


#Также создадим вспомогательную функцию, которая будет применяться к бачам при итерации по torch.utils.data.DataLoader. Она поможет избежать #ошибок c размерностями внутри бачей.
def collate_fn(batch):
    return tuple(zip(*batch))