import torch
import numpy as np
import cv2
import torchvision.transforms as transforms

inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.255]
            )
sm = torch.nn.Softmax(dim=1)


def get_vis(images, output, target):

    res = []
    for t in range(images.shape[0]):
        in_dbg = inv_normalize(images[t]).cpu().numpy().transpose(1,2,0)
        in_dbg *= 255
        in_dbg = np.clip(in_dbg, 0, 255)
        in_dbg = in_dbg.astype(np.uint8)

        out_dbg = sm(output.detach()).cpu().numpy()[t].transpose(1,2,0)[:, :, 1]
        out_dbg -= out_dbg.min()
        out_dbg /= out_dbg.max()
        out_dbg *= 255
        out_dbg = out_dbg.astype(np.uint8)

        in_dbg1 = cv2.cvtColor(in_dbg.copy(), cv2.COLOR_RGB2RGBA)
        in_dbg1[:, :, 3] = out_dbg

        out_gt = target.cpu().numpy()[t].astype(np.float32)
        out_gt -= out_gt.min()
        out_gt /= out_gt.max()
        out_gt *= 255
        out_gt = out_gt.astype(np.uint8)

        res.append((t, (in_dbg, out_dbg, out_gt, in_dbg1)))

    return res
