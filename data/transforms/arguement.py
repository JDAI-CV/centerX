import cv2
import numpy as np
import random
import torch
import imgaug as ia
import imgaug.augmenters as iaa
import copy

# points = [
#    [(10.5, 20.5)],  # points on first image
#    [(50.5, 50.5), (60.5, 60.5), (70.5, 70.5)]  # points on second image
# ]
# image = cv2.imread('000000472375.jpg')
# inp_bbox = [np.array([124.71,196.18,124.71+372.85,196.18+356.81])]


'''
points = np.array([[ 80.90703725, 126.08039874,   0.        ],
 [ 72.72988313, 127.2840341,    0.        ],
 [ 86.29191076, 160.56158147,   0.        ],
 [ 80.87585772, 159.50228059,   0.        ],
 [ 81.09376061, 190.41214379,   0.        ],
 [ 77.63778624, 192.15852308,   0.        ],
 [ 84.55893103, 190.83034651,   0.        ],
 [ 88.24699688, 192.76283703,  0.        ],
 [ 70.1611101,  235.95892525,   0.        ],
 [106.62995965, 239.87347792,   0.        ],
 [ 66.48005009, 286.62669707,   0.        ],
 [128.05848894, 280.34743948,   0.        ]])

image = cv2.imread('demo.jpg')



def show(image,points):
    for i in points:
        cv2.circle(image,(int(i[0]), int(i[1])), 5, (0,255,0), -1)
    return image
'''


# def arguementation(image, dataset_dict, p=0.5):
def arguementation(image, p=0.5):
    if random.random() > p:
        return image  # ,dataset_dict

    # H,W,C = image.shape
    # inp_bbox = [anno['bbox'] for anno in dataset_dict['annotations']]
    # ia_bbox = []
    # for bbox in inp_bbox:
    #     tmp_bbox = ia.BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])
    #     ia_bbox.append(tmp_bbox)
    # ia_bbox = [ia_bbox]

    images = np.array([image]).astype(np.uint8)
    # image = random_flip(image)
    # image = random_scale(image)
    # image = random_angle_rotate(image)
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image
    # All augmenters with per_channel=0.5 will sample one value _per image_
    # in 50% of all cases. In all other cases they will sample new values
    # _per channel_.

    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            # iaa.Fliplr(0.5), # horizontally flip 50% of all images
            # iaa.Flipud(0.2), # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            # iaa.CropAndPad(
            #     percent=(-0.3, 0.3),
            #     pad_mode='constant',
            #     pad_cval=(0, 0)
            # ),
            # iaa.Affine(
            #     scale={"x": (0.6, 1.4), "y": (0.6, 1.4)},
            #     # scale images to 80-120% of their size, individually per axis
            #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
            #     fit_output=False,  # True
            #     order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            #     cval=(0, 0),  # if mode is constant, use a cval between 0 and 255
            #     mode='constant'  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            # ),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                       [
                           sometimes(iaa.Superpixels(p_replace=(0, 0.1), n_segments=(200, 300))),
                           # convert images into their superpixel representation
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                               iaa.AverageBlur(k=(2, 4)),
                               # blur image using local means with kernel sizes between 2 and 7
                               iaa.MedianBlur(k=(1, 5)),
                               # blur image using local medians with kernel sizes between 2 and 7
                           ]),
                           iaa.Sharpen(alpha=(0, 0.75), lightness=(0.1, 1.9)),  # sharpen images
                           iaa.Emboss(alpha=(0, 0.75), strength=(0, 1.0)),  # emboss images
                           # search either for all edges or for directed edges,
                           # blend the result with the original image using a blobby mask
                           # iaa.SimplexNoiseAlpha(iaa.OneOf([
                           #    iaa.EdgeDetect(alpha=(0, 0.25)),
                           #    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                           # ])),
                           # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                           # iaa.OneOf([
                           #    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                           #    #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                           # ]),
                           # iaa.Invert(0.05, per_channel=True), # invert color channels
                           iaa.Add((-20, 20), per_channel=0.5),
                           # change brightness of images (by -10 to 10 of original value)
                           # iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                           # either change the brightness of the whole image (sometimes
                           # per channel) or change the brightness of subareas
                           iaa.OneOf([
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),
                               # iaa.FrequencyNoiseAlpha(
                               #    exponent=(-4, 0),
                               #    first=iaa.Multiply((0.5, 1.5), per_channel=True),
                               #    second=iaa.LinearContrast((0.5, 2.0))
                               # )
                           ]),
                           iaa.LinearContrast((0.75, 1.5), per_channel=0.5),  # improve or worsen the contrast
                           iaa.Grayscale(alpha=(0.0, 0.3)),
                           # sometimes(iaa.ElasticTransformation(alpha=(0.5, 1.5), sigma=0.25)), # move pixels locally around (with random strengths)
                           # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.03))), # sometimes move parts of the image around
                           sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                       ],
                       random_order=True
                       )
        ],
        random_order=True
    )
    # images_aug, bbox_aug = seq(images=images, bounding_boxes=ia_bbox)
    images_aug = seq(images=images)
    # for k, bbox in enumerate(bbox_aug[0]):
    #     dataset_dict['annotations'][k]['bbox'][0] = max(min(bbox.x1,W-1),0)
    #     dataset_dict['annotations'][k]['bbox'][1] = max(min(bbox.y1,H-1),0)
    #     dataset_dict['annotations'][k]['bbox'][2] = max(min(bbox.x2,W-1),0)
    #     dataset_dict['annotations'][k]['bbox'][3] = max(min(bbox.y2,H-1),0)

    # image = show(image,keypoints)
    # cv2.imwrite('source.jpg',image)
    # for k,i in enumerate(points_aug[0]):
    #    keypoints[k,0] = i[0]
    #    keypoints[k,1] = i[1]
    # image_a = show(images_aug[0],keypoints)
    # cv2.imwrite('result.jpg',image_a)
    # images_aug_tensor_list = [torch.from_tensor(image).type(_dtype) for image in images_aug]
    return images_aug[0]  # , dataset_dict

# rst_image,bbox = arguementation(image,inp_bbox)
# cv2.rectangle(rst_image,(int(bbox[0][0]),int(bbox[0][1])),(int(bbox[0][2]),int(bbox[0][3])),(0,255,0),2)
# cv2.imwrite('demo.jpg',rst_image)
# print(image.shape,rst_image.shape)
