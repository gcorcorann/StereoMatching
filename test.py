import cv2
import numpy as np
import random

max_disp = 128

for i in range(194):
    left = cv2.imread(
            'data/kitti2012/training/image_1/000{:03}_10.png'.format(i), 0
            )
    right = cv2.imread(
            'data/kitti2012/training/image_0/000{:03}_10.png'.format(i), 0
            )
    disp = cv2.imread(
            'data/kitti2012/training/disp_noc/000{:03}_10.png'.format(i), 0
            )
    
    # list of non-zero indices
    disparities = np.nonzero(disp)
    r = random.randint(0, len(disparities[0])-1)
    
    p_y, p_x = disparities[0][-1], disparities[1][-1]
    d = disp[p_y, p_x]
    left_padded = np.zeros((9, 9))
    left_patch = left[p_y-4: p_y+4+1, p_x-d-4: p_x-d+4+1]
    left_padded[:left_patch.shape[0], :left_patch.shape[1]] = left_patch
    
    right_padded = np.zeros((9, 136))
    right_patch = right[p_y-4: p_y+4+1, p_x-4: p_x+4+max_disp]
    right_padded[:right_patch.shape[0], :right_patch.shape[1]] = right_patch
    
    if left_patch.shape != (9, 9):
        print('left:', left.shape)
        print('right:', right.shape)
        print('disp:', disp.shape)
        print('p_x:', p_x, 'p_y:', p_y)
        print(left_patch)
        print(left_padded)
        print('left_patch:', left_patch.shape)
        print('left_padded:', left_padded.shape)
        print('right_patch:', right_patch.shape)
        print('right_padded:', right_padded.shape)
        print('disparity:', d)
    
        cv2.rectangle(left, (p_x-d-4, p_y-4), (p_x-d+4, p_y+4),
                (255,255,255), 0)
        cv2.rectangle(right, (p_x-4, p_y-4), (p_x+4+max_disp, p_y+4+1),
                (255,255,255), 0)
        cv2.rectangle(disp, (p_x-5, p_y-5), (p_x+5, p_y+5), (255,255,255), 0)
        Display = np.vstack((left, right, disp))
        cv2.imshow('Display', Display)
        cv2.waitKey(0)
