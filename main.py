#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter, sobel

def fill_holes_bw(im):
    inv = cv2.bitwise_not(im)
    contours, _ = cv2.findContours(inv,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(inv,[contour],0,255,-1)
    return cv2.bitwise_not(inv)

def create_tree_mask(im):
    BOX_RESOLUTION = 25
    DETECTION_RATIO = 4.5

    map_filter = cv2.Canny((im),50,150)
    for y in range(BOX_RESOLUTION,map_filter.shape[0]+1,BOX_RESOLUTION):
        for x in range(BOX_RESOLUTION,map_filter.shape[1]+1,BOX_RESOLUTION):
            patch = map_filter[y-BOX_RESOLUTION:y,x-BOX_RESOLUTION:x]
            if np.sum(patch == 255) > int((BOX_RESOLUTION ** 2) / DETECTION_RATIO):
                # cv2.rectangle(map_filter, (x-50,y-50),(x,y),(255,0,0), 2)
                map_filter[y-BOX_RESOLUTION:y,x-BOX_RESOLUTION:x] = 0
            else:
                map_filter[y-BOX_RESOLUTION:y,x-BOX_RESOLUTION:x] = 255
    return cv2.bitwise_not(fill_holes_bw(map_filter))

def approx_center(im):

    TEMPLATE_SIZE = (800,800)
    TEMPLATE_CENTER = tuple(int(x/2) for x in TEMPLATE_SIZE)

    circles = np.zeros(TEMPLATE_SIZE,np.uint8)
    mask = np.zeros(TEMPLATE_SIZE,np.uint8)
    cv2.circle(mask, TEMPLATE_CENTER, radius=300, color=255, thickness=250)
    for radius in range(25,400,5):
        cv2.circle(circles, TEMPLATE_CENTER, radius=radius, color=255, thickness=1)
    res = cv2.matchTemplate(im, circles, cv2.TM_SQDIFF,mask=mask)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(res)
    (x, y) = maxLoc
    x += TEMPLATE_CENTER[0]
    y += TEMPLATE_CENTER[1]
    
    return (x,y)

def count_rings(patch):
    return np.sum([1 for x in patch if x == 255])

def approx_age(im, center):
    total = 0
    c = 0
    for x in range(-20,20):
        patch_x0 = im[0:center[0],center[1]+x]
        patch_y0 = im[center[0]+x,0:center[1]]
        patch_x1 = im[center[0]:2000,center[1]+x]
        patch_y1 = im[center[0]+x,center[1]:1500]

        total += count_rings(patch_x0)
        total += count_rings(patch_y0)
        total += count_rings(patch_x1)
        total += count_rings(patch_y1)
        c += 4

    age = total/c 

    cv2.line(im,(0,center[1]),(2000,center[1]),color=255, thickness=1)
    cv2.line(im,(center[0],0),(center[0],1500),color=255, thickness=1)
    plt.figure(figsize=(16,12))
    plt.imshow((im), cmap="gray")
    plt.show()

    return age



def main():
    im = cv2.resize(cv2.imread("input.tif"), (2000,1500), interpolation=cv2.INTER_LANCZOS4)

    filtered = cv2.medianBlur(im, 5)
    denoised = cv2.fastNlMeansDenoisingColored(filtered, h=10)
    unsharp_mask = denoised - gaussian_filter(denoised, sigma=2)
    sharpened = denoised + 1 * unsharp_mask

    edges = cv2.Canny((sharpened),50,125)

    tree_mask = create_tree_mask(filtered)
    tree_edges = cv2.bitwise_and(edges,tree_mask)
    tree = cv2.bitwise_and(im, im, mask=tree_mask)

    center = approx_center(edges)

    age = approx_age(tree_edges, center)
    print(age)
    
    cv2.line(tree,(0,center[1]),(2000,center[1]),color=255, thickness=1)
    cv2.line(tree,(center[0],0),(center[0],1500),color=255, thickness=1)
    plt.figure(figsize=(16,12))
    plt.imshow(tree, cmap="gray")
    plt.show()

if __name__ == "__main__":
    main()