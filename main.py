#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter

MASK_MEDIAN_BLUR_SIZE = 3
BOX_RESOLUTION = 25
DETECTION_RATIO = 4

def fill_holes_bw(im):
    inv = cv2.bitwise_not(im)
    contours, _ = cv2.findContours(inv,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(inv,[contour],0,255,-1)
    return cv2.bitwise_not(inv)

def erode(im):
    erode_kernel = np.zeros((BOX_RESOLUTION*3,BOX_RESOLUTION*3),np.uint8)
    erode_kernel[BOX_RESOLUTION:2*BOX_RESOLUTION+1, ::] = 255
    erode_kernel[::, BOX_RESOLUTION:2*BOX_RESOLUTION+1] = 255
    
    cv2.imwrite('./doc/erode_kernel.png', erode_kernel)

    return cv2.erode(im,erode_kernel)

### Create a black/white mask to isolate the tree in the image.
### Canny Edge Detection is used on the original image (filtered a bit with median filter) to draw all edges.
### The program assumes that there are more sharp edges on the tree than in the background (manual blurring of the background in the input image can help)
### In a grid with the size of BOX_RESOLUTION all edge-pixels are counted. Is the count over a certain threshold (BOX_RESOLUTION^2 / DETECTION_RATIO)
### the grid gets marked as containing the tree.
### Afterwards all holes in the mask get filled and some stray boxes on the edges get eroded.
def create_tree_mask(im):
    mask = cv2.Canny(cv2.medianBlur(im, MASK_MEDIAN_BLUR_SIZE),50,150)
    for y in range(BOX_RESOLUTION,mask.shape[0]+1,BOX_RESOLUTION):
        for x in range(BOX_RESOLUTION,mask.shape[1]+1,BOX_RESOLUTION):
            patch = mask[y-BOX_RESOLUTION:y,x-BOX_RESOLUTION:x]
            if np.sum(patch == 255) > int((BOX_RESOLUTION ** 2) / DETECTION_RATIO):
                # cv2.rectangle(map_filter, (x-50,y-50),(x,y),(255,0,0), 2)
                mask[y-BOX_RESOLUTION:y,x-BOX_RESOLUTION:x] = 0
            else:
                mask[y-BOX_RESOLUTION:y,x-BOX_RESOLUTION:x] = 255
    cv2.imwrite('./doc/mask_before_filling.png', mask)
    
    mask = cv2.bitwise_not(fill_holes_bw(mask))
    cv2.imwrite('./doc/mask_before_erode.png', mask)
     
    mask = erode(mask)
    cv2.imwrite('./doc/final_mask.png', mask)

    return mask

def approx_center(im):
    TEMPLATE_SIZE = (800,800)
    TEMPLATE_CENTER = tuple(int(x/2) for x in TEMPLATE_SIZE)

    circles = np.zeros(TEMPLATE_SIZE,np.uint8)
    for radius in range(25,400,5):
        cv2.circle(circles, TEMPLATE_CENTER, radius=radius, color=255, thickness=1)
    cv2.imwrite('./doc/sqdiff_template.png', circles)
    
    mask = np.zeros(TEMPLATE_SIZE,np.uint8)
    cv2.circle(mask, TEMPLATE_CENTER, radius=300, color=255, thickness=250)
    cv2.imwrite('./doc/sqdiff_mask.png', mask)
    
    res = cv2.matchTemplate(im, circles, cv2.TM_SQDIFF,mask=mask)
    (_, _, _, max_loc) = cv2.minMaxLoc(res)
    (x, y) = max_loc
    x += TEMPLATE_CENTER[0]
    y += TEMPLATE_CENTER[1]
    
    im_tmp = im.copy()
    cv2.rectangle(im_tmp,max_loc,(max_loc[0]+800,max_loc[1]+800),color=255,thickness=2)
    cv2.imwrite('./doc/matched_template.png', im_tmp)
    
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

    cv2.line(im,(0,center[1]),(2000,center[1]),color=255, thickness=3)
    cv2.line(im,(center[0],0),(center[0],1500),color=255, thickness=3)
    
    cv2.imwrite('./doc/counting_cross_section_edges_and_masked.png', im)

    # plt.figure(figsize=(16,12))
    # plt.imshow((im), cmap="gray")
    # plt.show()

    return age



def main():
    im = cv2.resize(cv2.imread("input4.tif"), (2000,1500), interpolation=cv2.INTER_LANCZOS4)

    filtered = cv2.medianBlur(im, 5)
    cv2.imwrite('./doc/median_filtered.png', filtered)
    denoised = cv2.fastNlMeansDenoisingColored(filtered, h=8, templateWindowSize=5,searchWindowSize=15)
    cv2.imwrite('./doc/nlmeans_denoised.png', denoised)
    unsharp_mask = denoised - gaussian_filter(denoised, sigma=2)
    sharpened = denoised + 1 * unsharp_mask
    cv2.imwrite('./doc/sharpened.png', sharpened)

    edges = cv2.Canny(sharpened,50,125)
    cv2.imwrite('./doc/canny_edges.png', edges)

    tree_mask = create_tree_mask(im)
    tree_edges = cv2.bitwise_and(edges,tree_mask)
    tree = cv2.bitwise_and(im, im, mask=tree_mask)

    center = approx_center(tree_edges)

    age = approx_age(tree_edges, center)
    print(f"Estimated age: {age} years")
    
    cv2.line(tree,(0,center[1]),(2000,center[1]),color=255, thickness=1)
    cv2.line(tree,(center[0],0),(center[0],1500),color=255, thickness=1)
    cv2.imwrite('./doc/original_with_mask_and_cross_section.png', tree)

if __name__ == "__main__":
    main()