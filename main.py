import collections
import os

import cv2
import torch
from torch.utils.data import DataLoader
from vit_pytorch import ViT, Dino
from vit_pytorch.recorder import Recorder
import numpy as np

from dataset import Dataset
from settings import settings


def stack_to_original(orig, im):
    orig = orig.transpose((1, 2, 0)) / 256
    im = im.transpose((1, 2, 0))
    im = np.stack((im, im, im), axis=2).squeeze(-1)
    return np.hstack((orig[:257, :257, :], im))


if __name__ == '__main__':
    device = settings.device
    model_weights_path = os.path.join(settings.model_weights_folder, 'pretrained-net.pt')

    model = ViT(
        image_size=settings.image_size,
        patch_size=settings.patch_size,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=8,
        mlp_dim=2048
    ).to(device)
    # model = Recorder(model)
    try:
        model.load_state_dict(torch.load(model_weights_path, map_location='cpu'), strict=False)
    except Exception:
        pass

    learner = Dino(
        model,
        image_size=settings.image_size,
        hidden_layer=-2,  # hidden layer name or index, from which to extract the embedding | 'to_latent'
        projection_hidden_size=256,  # projector network hidden dimension
        projection_layers=4,  # number of layers in projection network
        num_classes_K=65336,  # output logits dimensions (referenced as K in paper)
        student_temp=0.9,  # student temperature
        teacher_temp=0.04,  # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
        local_upper_crop_scale=0.4,  # upper bound for local crop - 0.4 was recommended in the paper
        global_lower_crop_scale=0.5,  # lower bound for global crop - 0.5 was recommended in the paper
        moving_average_decay=0.95,  # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
        center_moving_average_decay=0.95,
        # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok
    ).to(device)

    opt = torch.optim.Adam(learner.parameters(), lr=settings.lr)

    dataset = Dataset(images_dir='data/', images_size=settings.image_size)

    if settings.train:
        train_loader = DataLoader(dataset, shuffle=True, batch_size=settings.train_batch_size, num_workers=4)
        losses_queue = collections.deque()
        for i in range(1, settings.epochs + 1):
            for batch_idx, data in enumerate(train_loader):
                data = data.to(device)
                loss = learner(data)
                losses_queue.append(loss.item())
                opt.zero_grad()
                loss.backward()
                opt.step()
                learner.update_moving_average()  # update moving average of teacher encoder and teacher centers
            print(f'epoch: {i} - loss: {sum(losses_queue) / len(losses_queue)}')
            losses_queue.clear()
        torch.save(model.state_dict(), model_weights_path)

    if not settings.train:
        model = Recorder(model)
        test_loader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=1)
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            preds, attns = model(data)
            # attns.shape = (batch x layers x heads x patch x patch)
            head = attns.detach().numpy()[0, -1, :, :, :]
            for i in range(head.shape[0]):
                cv2.imshow('win', stack_to_original(data[0, ...].detach().numpy(), head[[i], ...]))
                cv2.waitKey(0)