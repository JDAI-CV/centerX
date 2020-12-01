import torch
from torch.nn import functional as F

def batch_padding(batch_images,
                  div=32,
                  pad_value: float = 0.0):
    max_size = (
        # In tracing mode, x.shape[i] is Tensor, and should not be converted
        # to int: this will cause the traced graph to have hard-coded shapes.
        # Instead we should make max_size a Tensor that depends on these tensors.
        # Using torch.stack twice seems to be the best way to convert
        # list[list[ScalarTensor]] to a Tensor
        torch.stack(
            [
                torch.stack([torch.as_tensor(dim) for dim in size])
                for size in [tuple(img.shape) for img in batch_images]
            ]
        )
            .max(0)
            .values
    )

    if div > 1:
        stride = div
        # the last two dims are H,W, both subject to divisibility requirement
        max_size = torch.cat([max_size[:-2], (max_size[-2:] + (stride - 1)) // stride * stride])

    image_sizes = [tuple(im.shape[-2:]) for im in batch_images]

    if len(batch_images) == 1:
        # This seems slightly (2%) faster.
        # TODO: check whether it's faster for multiple images as well
        image_size = image_sizes[0]
        padding_size = [0, max_size[-1] - image_size[1], 0, max_size[-2] - image_size[0]]
        if all(x == 0 for x in padding_size):  # https://github.com/pytorch/pytorch/issues/31734
            batched_imgs = batch_images[0].unsqueeze(0)
        else:
            padded = F.pad(batch_images[0], padding_size, value=pad_value)
            batched_imgs = padded.unsqueeze_(0)
    else:
        # max_size can be a tensor in tracing mode, therefore use tuple()
        batch_shape = (len(batch_images),) + tuple(max_size)
        batched_imgs = batch_images[0].new_full(batch_shape, pad_value)
        for img, pad_img in zip(batch_images, batched_imgs):
            pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)

    return batched_imgs