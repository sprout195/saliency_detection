import cv2
import numpy as np
import random


def get_cv2_func(mode):
    if mode == 0:
        print('Run with StaticSaliencySpectralResidual!')
        return cv2.saliency.StaticSaliencySpectralResidual_create()
    elif mode == 1:
        print('Run with StaticSaliencyFineGrained!')
        return cv2.saliency.StaticSaliencyFineGrained_create()
    elif mode == 2:
        return None



def SpectralResidual(cv2_func, image, ratio, model_name='c3d'):

    if cv2_func:
        (success, saliencyMap) = cv2_func.computeSaliency(image)
        flat_saliency = saliencyMap.flatten()
        MASK = np.zeros_like(saliencyMap)
        flat_MASK = MASK.flatten()
        indices = np.argsort(-flat_saliency)

        useful_indices = indices[: int(len(indices) * ratio)]
        flat_MASK[useful_indices] = 1
        MASK = np.reshape(flat_MASK, saliencyMap.shape)
        MASK = np.stack([MASK, MASK, MASK], axis=2)

    else:
        print('Run with Random')
        MASK = RandomSpatial(image, ratio, model_name)
    return MASK


def RandomSpatial(image, ratio, model_name):
    MASK = np.zeros_like(image)
    flat_MASK = MASK.flatten()
    random.seed(1024)
    indices   = random.sample([i for i in range(len(flat_MASK))], int(len(flat_MASK) * ratio))
    flat_MASK[indices] = 1
    MASK = np.reshape(flat_MASK, image.shape)
    MASK = torch.from_numpy(MASK)
    return MASK



def initialize_salient_region_mask(self):
    cv2_func = get_cv2_func(self.spatial_mode)
    MASKs = []
    for i in range(self.seq_len):
        if self.seq_axis == 1:
            tmp_image = self.image_ori[:, i, :, :]
        else:
            tmp_image = self.image_ori[i]
        this_mask = SpectralResidual(cv2_func, tmp_image.cpu(), self.spatial_ratio, self.model_name)
        MASKs.append(this_mask)
    # [seq_len, height, width, num_channels]
    if self.model_name == 'c3d':
        MASKs = torch.stack(MASKs).permute(3, 0, 1, 2).cuda()
    elif self.model_name == 'lrcn':
        MASKs = torch.stack(MASKs).cuda()
    elif self.model_name == 'flownet':
        MASKs = torch.stack(MASKs).permute(0, 3, 1, 2).cuda()
    self.MASK = MASKs
