import numpy as np
import cv2
from scipy.ndimage import rotate

def apply_filter_a(src:np.ndarray):
    src_copy = np.copy(src)
    print(f"Input image shape: {src_copy.shape}, dtype: {src_copy.dtype}")  
    f1 = np.array([[[ 0,  0,  0,  0,  0],
        [ 0,  1,  0,  0,  0],
        [ 0,  0, -1,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0]],

       [[ 0,  0,  0,  0,  0],
        [ 0,  0,  1,  0,  0],
        [ 0,  0, -1,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0]],

       [[ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  1,  0],
        [ 0,  0, -1,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0]],

       [[ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0, -1,  1,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0]],

       [[ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0, -1,  0,  0],
        [ 0,  0,  0,  1,  0],
        [ 0,  0,  0,  0,  0]],

       [[ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0, -1,  0,  0],
        [ 0,  0,  1,  0,  0],
        [ 0,  0,  0,  0,  0]],

       [[ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0, -1,  0,  0],
        [ 0,  1,  0,  0,  0],
        [ 0,  0,  0,  0,  0]],

       [[ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  1, -1,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0]]])
    
    img = cv2.filter2D(src=src_copy, kernel=f1[0], ddepth=-1)
    for filter in f1[1:]:
        img = cv2.add(img,cv2.filter2D(src=src_copy, kernel=filter, ddepth=-1))

    return img//8

def apply_filter_b(src:np.ndarray):
    src_copy = np.copy(src)
    f2 = np.array([[[ 0,  0,  0,  0,  0],
                    [ 0,  2,  1,  0,  0],
                    [ 0,  1, -3,  0,  0],
                    [ 0,  0,  0,  1,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0, -1,  0,  0],
                    [ 0,  0,  3,  0,  0],
                    [ 0,  0, -3,  0,  0],
                    [ 0,  0,  1,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  1,  2,  0],
                    [ 0,  0, -3,  1,  0],
                    [ 0,  1,  0,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  1, -3,  3, -1],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  1,  0,  0,  0],
                    [ 0,  0, -3,  1,  0],
                    [ 0,  0,  1,  2,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  1,  0,  0],
                    [ 0,  0, -3,  0,  0],
                    [ 0,  0,  3,  0,  0],
                    [ 0,  0, -1,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  1,  0],
                    [ 0,  1, -3,  0,  0],
                    [ 0,  2,  1,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0],
                    [-1,  3, -3,  1,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0]]])

    img = cv2.filter2D(src=src_copy, kernel=f2[0], ddepth=-1)
    for filter in f2[1:]:
        img = cv2.add(img,cv2.filter2D(src=src_copy, kernel=filter, ddepth=-1))

    return img//8



def apply_filter_c(src:np.ndarray):
    src_copy=np.copy(src)
    f3 = np.array([[[ 0,  0,  0,  0,  0],
                    [ 0,  0,  1,  0,  0],
                    [ 0,  0, -2,  0,  0],
                    [ 0,  0,  1,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  1, -2,  1,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  1,  0,  0,  0],
                    [ 0,  0, -2,  0,  0],
                    [ 0,  0,  0,  1,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  1,  0],
                    [ 0,  0, -2,  0,  0],
                    [ 0,  1,  0,  0,  0],
                    [ 0,  0,  0,  0,  0]]])
    
    img = cv2.filter2D(src=src_copy, kernel=f3[0], ddepth=-1)
    for filter in f3[1:]:
        img = cv2.add(img,cv2.filter2D(src=src_copy, kernel=filter, ddepth=-1))

    return img//4


def apply_filter_d(src:np.ndarray):
    src_copy=np.copy(src)
    f4 = np.array([[[ 0,  0,  0,  0,  0],
                    [ 0, -1,  2, -1,  0],
                    [ 0,  2, -4,  2,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0, -1,  2,  0,  0],
                    [ 0,  2, -4,  0,  0],
                    [ 0, -1,  2,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  2, -4,  2,  0],
                    [ 0, -1,  2, -1,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  2, -1,  0],
                    [ 0,  0, -4,  2,  0],
                    [ 0,  0,  2, -1,  0],
                    [ 0,  0,  0,  0,  0]]])

    img = cv2.filter2D(src=src_copy, kernel=f4[0], ddepth=-1)
    for filter in f4[1:]:
        img = cv2.add(img,cv2.filter2D(src=src_copy, kernel=filter, ddepth=-1))

    return img//4

def apply_filter_e(src:np.ndarray):
    src_copy=np.copy(src)
    f5 = np.array([[[  1,   2,  -2,   2,   1],
                    [  2,  -6,   8,  -6,   2],
                    [ -2,   8, -12,   8,  -2],
                    [  0,   0,   0,   0,   0],
                    [  0,   0,   0,   0,   0]],

                [[  1,   2,  -2,   0,   0],
                    [  2,  -6,   8,   0,   0],
                    [ -2,   8, -12,   0,   0],
                    [  2,  -6,   8,   0,   0],
                    [  1,   2,  -2,   0,   0]],

                [[  0,   0,   0,   0,   0],
                    [  0,   0,   0,   0,   0],
                    [ -2,   8, -12,   8,  -2],
                    [  2,  -6,   8,  -6,   2],
                    [  1,   2,  -2,   2,   1]],

                [[  0,   0,  -2,   2,   1],
                    [  0,   0,   8,  -6,   2],
                    [  0,   0, -12,   8,  -2],
                    [  0,   0,   8,  -6,   2],
                    [  0,   0,  -2,   2,   1]]])
    
    img = cv2.filter2D(src=src_copy, kernel=f5[0], ddepth=-1)
    for filter in f5[1:]:
        img=cv2.add(img,cv2.filter2D(src=src_copy, kernel=filter, ddepth=-1))

    return img//4

def apply_filter_f(src:np.ndarray):
    src_copy=np.copy(src)
    f5 = np.asarray([[ 0,  0,  0,  0,  0],
                    [ 0,  -1,  2, -1,  0],
                    [ 0,  2,  -4,  2,  0],
                    [ 0,  -1,  2, -1,  0],
                    [ 0,  0,  0,  0,  0]])
    
    img = cv2.filter2D(src=src_copy, kernel=f5, ddepth=-1)
    return img


def apply_filter_g(src:np.ndarray):
    src_copy=np.copy(src)
    f5 = np.asarray([[ -1,   2,  -2,   2,  -1],
                    [  2,  -6,   8,  -6,   2],
                    [ -2,   8, -12,   8,  -2],
                    [  2,  -6,   8,  -6,   2],
                    [ -1,   2,  -2,   2,  -1]])
    
    img = cv2.filter2D(src=src_copy, kernel=f5, ddepth=-1)
    return img

def apply_all_filters(src: np.ndarray):  
    if not isinstance(src, np.ndarray) or len(src.shape) != 4:  
        raise ValueError("Input should be a batch of images with shape (B, C, H, W).")  

    batch_size, channels, height, width = src.shape  
    
    combined_img = np.zeros((batch_size, height, width), dtype=np.float32)  

    for i in range(batch_size):  
        img_copy = np.copy(src[i])
        
        filtered_a = apply_filter_a(img_copy.transpose(1, 2, 0))  # (H, W, C)  
        filtered_b = apply_filter_b(img_copy.transpose(1, 2, 0))  
        filtered_c = apply_filter_c(img_copy.transpose(1, 2, 0))  
        filtered_d = apply_filter_d(img_copy.transpose(1, 2, 0))  
        filtered_e = apply_filter_e(img_copy.transpose(1, 2, 0))  
        filtered_f = apply_filter_f(img_copy.transpose(1, 2, 0))  
        filtered_g = apply_filter_g(img_copy.transpose(1, 2, 0))  

        filtered_combined = (filtered_a + filtered_b + filtered_c +  
                             filtered_d + filtered_e + filtered_f +  
                             filtered_g) / 7.0  

        img_gray = cv2.cvtColor(filtered_combined.astype(np.uint8), cv2.COLOR_RGB2GRAY)  

        img_thresh = np.median(img_gray) + 2  
        
        _, img_binary = cv2.threshold(img_gray, img_thresh, 255, cv2.THRESH_BINARY)  
        
        combined_img[i] = img_binary  # (H, W)  

    return combined_img 


import torch  
import torch.nn as nn  
import torch.nn.functional as F  

class FilterLayer(nn.Module):  
    def __init__(self):  
        super(FilterLayer, self).__init__()  
        
        self.filter_a = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, padding=2, bias=False)  
        self.filter_b = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, padding=2, bias=False)  
        self.filter_c = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=5, padding=2, bias=False)  
        self.filter_d = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=5, padding=2, bias=False)  
        self.filter_e = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=5, padding=2, bias=False)  
        self.filter_f = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5, padding=2, bias=False)  
        self.filter_g = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5, padding=2, bias=False)  

        self._init_filters()  
        for param in self.parameters():  
            param.requires_grad = False  

    def _init_filters(self):  
        # Weight initialization for filter a (8 input channels, 1 output channel, 5x5 kernel)  
        self.filter_a.weight.data = torch.tensor([[  
            [[ 0,  0,  0,  0,  0],  
            [ 0,  1,  0,  0,  0],  
            [ 0,  0, -1,  0,  0],  
            [ 0,  0,  0,  0,  0],  
            [ 0,  0,  0,  0,  0]],  
            [[ 0,  0,  0,  0,  0],  
            [ 0,  0,  1,  0,  0],  
            [ 0,  0, -1,  0,  0],  
            [ 0,  0,  0,  0,  0],  
            [ 0,  0,  0,  0,  0]],  
            [[ 0,  0,  0,  0,  0],  
            [ 0,  0,  0,  1,  0],  
            [ 0,  0, -1,  0,  0],  
            [ 0,  0,  0,  0,  0],  
            [ 0,  0,  0,  0,  0]],  
            [[ 0,  0,  0,  0,  0],  
            [ 0,  0,  0,  0,  0],  
            [ 0,  0, -1,  1,  0],  
            [ 0,  0,  0,  0,  0],  
            [ 0,  0,  0,  0,  0]],  
            [[ 0,  0,  0,  0,  0],  
            [ 0,  0,  0,  0,  0],  
            [ 0,  0, -1,  0,  0],  
            [ 0,  0,  0,  1,  0],  
            [ 0,  0,  0,  0,  0]],  
            [[ 0,  0,  0,  0,  0],  
            [ 0,  0,  0,  0,  0],  
            [ 0,  0, -1,  0,  0],  
            [ 0,  0,  1,  0,  0],  
            [ 0,  0,  0,  0,  0]],  
            [[ 0,  0,  0,  0,  0],  
            [ 0,  0,  0,  0,  0],  
            [ 0,  0, -1,  0,  0],  
            [ 0,  1,  0,  0,  0],  
            [ 0,  0,  0,  0,  0]],  
            [[ 0,  0,  0,  0,  0],  
            [ 0,  0,  0,  0,  0],  
            [ 0,  1, -1,  0,  0],  
            [ 0,  0,  0,  0,  0],  
            [ 0,  0,  0,  0,  0]]  
        ]], dtype=torch.float32)  # shape: (8, 3, 5, 5)  

        # Weight initialization for filter b (8 input channels, 1 output channel, 5x5 kernel)  
        self.filter_b.weight.data = torch.tensor([[  
            [[ 0,  0,  0,  0,  0],  
            [ 0,  2,  1,  0,  0],  
            [ 0,  1, -3,  0,  0],  
            [ 0,  0,  0,  1,  0],  
            [ 0,  0,  0,  0,  0]],  
            [[ 0,  0, -1,  0,  0],  
            [ 0,  0,  3,  0,  0],  
            [ 0,  0, -3,  0,  0],  
            [ 0,  0,  1,  0,  0],  
            [ 0,  0,  0,  0,  0]],  
            [[ 0,  0,  0,  0,  0],  
            [ 0,  0,  1,  2,  0],  
            [ 0,  0, -3,  1,  0],  
            [ 0,  1,  0,  0,  0],  
            [ 0,  0,  0,  0,  0]],  
            [[ 0,  0,  0,  0,  0],  
            [ 0,  0,  0,  0,  0],  
            [ 0,  1, -3,  3, -1],  
            [ 0,  0,  0,  0,  0],  
            [ 0,  0,  0,  0,  0]],  
            [[ 0,  0,  0,  0,  0],  
            [ 0,  1,  0,  0,  0],  
            [ 0,  0, -3,  1,  0],  
            [ 0,  0,  1,  2,  0],  
            [ 0,  0,  0,  0,  0]],  
            [[ 0,  0,  0,  0,  0],  
            [ 0,  0,  1,  0,  0],  
            [ 0,  0, -3,  0,  0],  
            [ 0,  0,  3,  0,  0],  
            [ 0,  0, -1,  0,  0]],  
            [[ 0,  0,  0,  0,  0],  
            [ 0,  0,  0,  1,  0],  
            [ 0,  1, -3,  0,  0],  
            [ 0,  2,  1,  0,  0],  
            [ 0,  0,  0,  0,  0]],  
            [[ 0,  0,  0,  0,  0],  
            [ 0,  0,  0,  0,  0],  
            [-1,  3, -3,  1,  0],  
            [ 0,  0,  0,  0,  0],  
            [ 0,  0,  0,  0,  0]]  
        ]], dtype=torch.float32)  # shape: (8, 3, 5, 5)  

        # Weight initialization for filter c (4 input channels, 1 output channel, 5x5 kernel)  
        self.filter_c.weight.data = torch.tensor([[  
            [[ 0,  0,  0,  0,  0],  
            [ 0,  0,  1,  0,  0],  
            [ 0,  0, -2,  0,  0],  
            [ 0,  0,  1,  0,  0],  
            [ 0,  0,  0,  0,  0]],  
            [[ 0,  0,  0,  0,  0],  
            [ 0,  0,  0,  0,  0],  
            [ 0,  1, -2,  1,  0],  
            [ 0,  0,  0,  0,  0],  
            [ 0,  0,  0,  0,  0]],  
            [[ 0,  0,  0,  0,  0],  
            [ 0,  1,  0,  0,  0],  
            [ 0,  0, -2,  0,  0],  
            [ 0,  0,  0,  1,  0],  
            [ 0,  0,  0,  0,  0]],  
            [[ 0,  0,  0,  0,  0],  
            [ 0,  0,  0,  1,  0],  
            [ 0,  0, -2,  0,  0],  
            [ 0,  1,  0,  0,  0],  
            [ 0,  0,  0,  0,  0]]  
        ]], dtype=torch.float32)  # shape: (4, 3, 5, 5)  

        # Weight initialization for filter d (4 input channels, 1 output channel, 5x5 kernel)  
        self.filter_d.weight.data = torch.tensor([[  
            [[ 0,  0,  0,  0,  0],  
            [ 0, -1,  2, -1,  0],  
            [ 0,  2, -4,  2,  0],  
            [ 0,  0,  0,  0,  0],  
            [ 0,  0,  0,  0,  0]],  
            [[ 0,  0,  0,  0,  0],  
            [ 0, -1,  2,  0,  0],  
            [ 0,  2, -4,  0,  0],  
            [ 0, -1,  2,  0,  0],  
            [ 0,  0,  0,  0,  0]],  
            [[ 0,  0,  0,  0,  0],  
            [ 0,  0,  0,  0,  0],  
            [ 0,  2, -4,  2,  0],  
            [ 0, -1,  2, -1,  0],  
            [ 0,  0,  0,  0,  0]],  
            [[ 0,  0,  0,  0,  0],  
            [ 0,  0,  2, -1,  0],  
            [ 0,  0, -4,  2,  0],  
            [ 0,  0,  2, -1,  0],  
            [ 0,  0,  0,  0,  0]]  
        ]], dtype=torch.float32)  # shape: (4, 3, 5, 5)  

        # Weight initialization for filter e (4 input channels, 1 output channel, 5x5 kernel)  
        self.filter_e.weight.data = torch.tensor([[  
            [[  1,   2,  -2,   2,   1],  
            [  2,  -6,   8,  -6,   2],  
            [ -2,   8, -12,   8,  -2],  
            [  0,   0,   0,   0,   0],  
            [  0,   0,   0,   0,   0]],  
            [[  1,   2,  -2,   0,   0],  
            [  2,  -6,   8,   0,   0],  
            [ -2,   8, -12,   0,   0],  
            [  2,  -6,   8,   0,   0],  
            [  1,   2,  -2,   0,   0]],  
            [[  0,   0,   0,   0,   0],  
            [  0,   0,   0,   0,   0],  
            [ -2,   8, -12,   8,  -2],  
            [  2,  -6,   8,  -6,   2],  
            [  1,   2,  -2,   2,   1]],  
            [[  0,   0,  -2,   2,   1],  
            [  0,   0,   8,  -6,   2],  
            [  0,   0, -12,   8,  -2],  
            [  0,   0,   8,  -6,   2],  
            [  0,   0,  -2,   2,   1]]  
        ]], dtype=torch.float32)  # shape: (4, 3, 5, 5)  

        # Weight initialization for filter f (1 input channel, 1 output channel, 5x5 kernel)  
        self.filter_f.weight.data = torch.tensor([[  
            [[ 0,  0,  0,  0,  0],  
            [ 0,  -1,  2, -1,  0],  
            [ 0,  2,  -4,  2,  0],  
            [ 0,  -1,  2, -1,  0],  
            [ 0,  0,  0,  0,  0]]  
        ]], dtype=torch.float32)  # shape: (1, 3, 5, 5)  

        # Weight initialization for filter g (1 input channel, 1 output channel, 5x5 kernel)  
        self.filter_g.weight.data = torch.tensor([[  
            [[ -1,   2,  -2,   2,  -1],  
            [  2,  -6,   8,  -6,   2],  
            [ -2,   8, -12,   8,  -2],  
            [  2,  -6,   8,  -6,   2],  
            [ -1,   2,  -2,   2,  -1]]  
        ]], dtype=torch.float32)  # shape: (1, 3, 5, 5)
        self.filter_a.weight.data = self.filter_a.weight.data.permute(1, 0, 2, 3)  
        self.filter_b.weight.data = self.filter_b.weight.data.permute(1, 0, 2, 3)  
        self.filter_c.weight.data = self.filter_c.weight.data.permute(1, 0, 2, 3)  
        self.filter_d.weight.data = self.filter_d.weight.data.permute(1, 0, 2, 3)  
        self.filter_e.weight.data = self.filter_e.weight.data.permute(1, 0, 2, 3)  
        self.filter_f.weight.data = self.filter_f.weight.data.permute(1, 0, 2, 3)  
        self.filter_g.weight.data = self.filter_g.weight.data.permute(1, 0, 2, 3)  

        self.filter_a.weight.data = self.filter_a.weight.data.repeat(1, 3, 1, 1) 
        self.filter_b.weight.data = self.filter_b.weight.data.repeat(1, 3, 1, 1)  
        self.filter_c.weight.data = self.filter_c.weight.data.repeat(1, 3, 1, 1)  
        self.filter_d.weight.data = self.filter_d.weight.data.repeat(1, 3, 1, 1)  
        self.filter_e.weight.data = self.filter_e.weight.data.repeat(1, 3, 1, 1)  
        self.filter_f.weight.data = self.filter_f.weight.data.repeat(1, 3, 1, 1)  
        self.filter_g.weight.data = self.filter_g.weight.data.repeat(1, 3, 1, 1)  

    def forward(self, x):  
        out_a = self.filter_a(x)  # (B, 8, H, W)  
        out_b = self.filter_b(x)  # (B, 8, H, W)  
        out_c = self.filter_c(x)  # (B, 4, H, W)  
        out_d = self.filter_d(x)  # (B, 4, H, W)  
        out_e = self.filter_e(x)  # (B, 4, H, W)  
        out_f = self.filter_f(x)  # (B, 1, H, W)  
        out_g = self.filter_g(x)  # (B, 1, H, W)  

        out_a_mean = out_a.mean(dim=1)  # (B, H, W)  
        out_b_mean = out_b.mean(dim=1)  # (B, H, W)  
        out_c_mean = out_c.mean(dim=1)  # (B, H, W)  
        out_d_mean = out_d.mean(dim=1)  # (B, H, W)  
        out_e_mean = out_e.mean(dim=1)  # (B, H, W)  
        out_f_mean = out_f.squeeze(1) 
        out_g_mean = out_g.squeeze(1) 
 
        img_gray = (out_a_mean + out_b_mean + out_c_mean + out_d_mean + out_e_mean + out_f_mean + out_g_mean) / 7.0  

        img_thresh = torch.median(img_gray.view(img_gray.size(0), -1), dim=1).values +2/255.0 
        img_binary = torch.zeros_like(img_gray, dtype=torch.float32)  
        for i in range(img_binary.size(0)): 
            img_binary[i] = (img_gray[i] > img_thresh[i].item()) 
            # print(img_binary[i])
        return img_binary  

# 示例使用  
if __name__ == "__main__":  
    import torch  
    import torchvision.transforms as transforms  
    from PIL import Image  
    import torchvision.utils as vutils  

    input_image = Image.open('/home/new_baseline/patchcraft/reconstructed_images/r_image_103182.png').convert('RGB')  # 替换为您的图片路径  
    transform = transforms.Compose([  
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),  
    ])  
    input_tensor = transform(input_image).unsqueeze(0).cuda()  
    input_tensor = input_tensor.repeat(2,1,1,1)
    filter_layer = FilterLayer().cuda()  

    output = filter_layer(input_tensor)  
    print(output.shape)
    output_image = output.cpu()  
    output_image_pil = transforms.ToPILImage()(output_image.cpu())    
    save_image_path = 'output_image.png'   
    output_image_path = 'output_image_pil.png' 
    output_image_pil.save(output_image_path)  

    print(f'Output image saved to {save_image_path}')  
    print(output.shape)  