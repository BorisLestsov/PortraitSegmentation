import numpy as np
import torch


def create_image_and_label(nx,ny,c, cnt = 10, r_min = 5, r_max = 50, border = 92, sigma = 20, rectangles=False):


    image = np.ones((nx, ny, c))
    label = np.zeros((nx, ny, 3), dtype=np.bool)
    mask = np.zeros((nx, ny), dtype=np.bool)
    for _ in range(cnt):
        a = np.random.randint(border, nx-border)
        b = np.random.randint(border, ny-border)
        r = np.random.randint(r_min, r_max)
        h = np.random.randint(1,255, size=(c,))

        y,x = np.ogrid[-a:nx-a, -b:ny-b]
        m = x*x + y*y <= r*r
        mask = np.logical_or(mask, m)

        image[m] = h

    label[mask, 1] = 1

    if rectangles:
        mask = np.zeros((nx, ny), dtype=np.bool)
        for _ in range(cnt//2):
            a = np.random.randint(nx)
            b = np.random.randint(ny)
            r =  np.random.randint(r_min, r_max)
            h = np.random.randint(1,255)

            m = np.zeros((nx, ny), dtype=np.bool)
            m[a:a+r, b:b+r] = True
            mask = np.logical_or(mask, m)
            image[m] = h

        label[mask, 2] = 1

        label[..., 0] = ~(np.logical_or(label[...,1], label[...,2]))

    image += np.random.normal(scale=sigma, size=image.shape)
    image -= np.amin(image)
    image /= np.amax(image)

    if rectangles:
        return image, label
    else:
        return image, label[..., 1]


class Ellipses(torch.utils.data.Dataset):
    channels = 3
    n_class = 2

    def __init__(self, nx, ny, transform=None, **kwargs):
        super(Ellipses, self).__init__()
        self.nx = nx
        self.ny = ny
        self.kwargs = kwargs

        self.transform = transform


    def __getitem__(self, index):
        img, label = create_image_and_label(self.nx, self.ny, c=3, **self.kwargs)

        if not self.transform is None:
            img = self.transform(img.astype(np.float32))
        label = torch.from_numpy(label.astype(np.int64))

        return img, label

    def __len__(self):
        return 100
