#!/usr/bin/env python3

import os

import cv2
import numpy as np

INPUT_FILE = "input.tif"

# Tree masking vars
MASK_MEDIAN_BLUR_SIZE = 3
BOX_SIZE = 25
DETECTION_RATIO = 4

# Template matching vars
TEMPLATE_SIZE = 850
TEMPLATE_SHAPE = (TEMPLATE_SIZE, TEMPLATE_SIZE)
TEMPLATE_CENTER = tuple(int(x/2) for x in TEMPLATE_SHAPE)

# Preprocessing vars
NLMEANS_STRENGTH = 10
SHARPEN_STRENGTH = 2


def fill_holes_bw(im):
    """Fills holes by taking the inverse of the mask and 
    drawing/filling contours (and inverting back).
    """
    inv = cv2.bitwise_not(im)
    contours, _ = cv2.findContours(
        inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(inv, [contour], 0, 255, -1)
    return cv2.bitwise_not(inv)


def erode(im):
    """Creates an eroding kernel/structure element (cross shaped, 
    see ./doc/erode_kernel.png) according to the BOX_SIZE.
    Then the images gets eroded by using this kernel.
    """
    erode_kernel = np.zeros((BOX_SIZE*3, BOX_SIZE*3), np.uint8)
    erode_kernel[BOX_SIZE:2*BOX_SIZE+1, ::] = 255
    erode_kernel[::, BOX_SIZE:2*BOX_SIZE+1] = 255

    cv2.imwrite('./doc/erode_kernel.png', erode_kernel)
    return cv2.erode(im, erode_kernel)


def create_tree_mask(im):
    """Create a black/white mask to isolate the tree in the image.
    Canny Edge Detection is used on the original image (filtered a bit 
    with median filter) to draw all edges. The program assumes that 
    there are more sharp edges on the tree than in the background 
    (manual blurring of the background in the input image can help).
    In a grid with the size of BOX_SIZE all edge-pixels are counted. 
    Is the count over a certain threshold (BOX_SIZE^2 / 
    DETECTION_RATIO) the grid gets marked as containing the tree.
    Afterwards all holes in the mask get filled and some stray boxes 
    on the edges get eroded.
    """
    mask = cv2.Canny(
        cv2.medianBlur(im, MASK_MEDIAN_BLUR_SIZE), 
        50, 
        150
    )
    for y in range(BOX_SIZE, mask.shape[0]+1, BOX_SIZE):
        for x in range(BOX_SIZE, mask.shape[1]+1, BOX_SIZE):
            patch = mask[y-BOX_SIZE:y, x-BOX_SIZE:x]
            if np.sum(patch == 255) > int((BOX_SIZE ** 2) / DETECTION_RATIO):
                # cv2.rectangle(map_filter, (x-50,y-50),(x,y),(255,0,0), 2)
                mask[y-BOX_SIZE:y, x-BOX_SIZE:x] = 0
            else:
                mask[y-BOX_SIZE:y, x-BOX_SIZE:x] = 255
    cv2.imwrite('./doc/mask_before_filling.png', mask)

    # invert mask again for the bitwise AND (line 154)
    mask = cv2.bitwise_not(fill_holes_bw(mask))
    cv2.imwrite('./doc/mask_before_erode.png', mask)

    mask = erode(mask)
    cv2.imwrite('./doc/final_mask.png', mask)
    return mask


def approx_center(im):
    """Approximates the center of the masked edges. 
    Takes an imaged preprocessed with the Canny edge detector and 
    masked via create_tree_mask() as input. Creates a template 
    filled with rings with increasing radii. Creates an additional 
    mask for the template matching. See ./doc/sqdiff_xxx.png for 
    visualization. Uses normed cross-correlation for template matching 
    https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html#gga3a7850640f1fe1f58fe91a2d7583695daf9c3ab9296f597ea71f056399a5831da.
    cv2.minMaxLoc() computes the maximum (among other) of the template 
    matching function. To get to the center, the center coordinates of 
    the template circle get added to the max-coordinates. In 
    ./doc/matched_template.png the matched template boundaries are 
    drawn for visualization.

    This method doesn't really work as it is thought because the rings 
    are far to irregular for the cross-correlation to get a good match 
    (if TEMPLATE_SIZE is chosen too small, the match can be very bad).
    On the upside, if one can adjust the TEMPLATE_SIZE accordingly to 
    the size of the tree stump, it can get some acceptable results.
    """
    circles = np.zeros(TEMPLATE_SHAPE, np.uint8)
    for radius in range(25, int(TEMPLATE_SIZE/2)+1, 10):
        cv2.circle(circles, TEMPLATE_CENTER,
                   radius=radius, color=255, thickness=1)
    cv2.imwrite('./doc/sqdiff_template.png', circles)

    mask = np.zeros(TEMPLATE_SHAPE, np.uint8)
    cv2.circle(
        mask, 
        TEMPLATE_CENTER,
        radius=int(TEMPLATE_SIZE/3),
        color=255, 
        thickness=int(TEMPLATE_SIZE/3)
    )
    cv2.imwrite('./doc/sqdiff_mask.png', mask)

    res = cv2.matchTemplate(
        im, 
        circles, 
        cv2.TM_CCORR_NORMED,
        mask=mask
    )
    (_, _, _, max_loc) = cv2.minMaxLoc(res)
    (x, y) = max_loc
    x += TEMPLATE_CENTER[0]
    y += TEMPLATE_CENTER[1]

    im_tmp = im.copy()
    cv2.rectangle(
        im_tmp, 
        max_loc, 
        (max_loc[0]+TEMPLATE_SIZE,max_loc[1]+TEMPLATE_SIZE), 
        color=255, 
        thickness=2
    )
    cv2.imwrite('./doc/matched_template.png', im_tmp)
    return (x, y)


def count_rings(patch):
    """Count the rings in a patch of the masked Canny edge detection. 
    I assume that for every ring on the three should be one edge.
    """
    return np.sum([1 for x in patch if x == 255])

def approx_age(im, center):
    """Approximate the age of the tree. 
    Takes the masked Canny edge detection and the calculated center 
    point. The deviates from the center by +-20 pixels on both axis 
    and takes the average of all to mitigate some irregularities. 
    See ./doc/counting_cross_section_edges_and_masked.png for 
    visualization of the cross-sections, along which the program 
    counts the edges/rings.
    """
    total = 0
    c = 0
    for deviation in range(-20, 20):
        patch_x0 = im[0:center[0], center[1]+deviation]
        patch_y0 = im[center[0]+deviation, 0:center[1]]
        patch_x1 = im[center[0]:2000, center[1]+deviation]
        patch_y1 = im[center[0]+deviation, center[1]:1500]

        total += count_rings(patch_x0)
        total += count_rings(patch_y0)
        total += count_rings(patch_x1)
        total += count_rings(patch_y1)
        c += 4

    age = total/c

    cv2.line(
        im, 
        (0, center[1]), 
        (2000, center[1]), 
        color=255, 
        thickness=3
    )
    cv2.line(
        im, 
        (center[0], 0), 
        (center[0], 1500), 
        color=255, 
        thickness=3
    )
    cv2.imwrite('./doc/counting_cross_section_edges_and_masked.png', im)
    return age


def main():
    """Main function of the program. 
    First step is to read the image and downscale (just a tiny bit) 
    to a nice and even resolution (Lanczos for good results).
    Then some basic image processing is done: 
    Median blur to eliminate hot/cold pixels
    --> non-local means denoising to remove even more noise 
    (adjust NLMEANS_STRENGTH if too much detail is lost or the image 
    stays too noisy)
    --> sharpening via unsharp mask to regain some detail after 
    denoising/blurring
    The processed image gets passed to Canny edge detection to create 
    a projection with all edges. This gets passed to create a mask for 
    the tree stump to isolate and remove the background (bitwise AND 
    on processed image / edges with mask). Then the isolated edges of 
    the tree stump get passed to center approximation. With the center 
    point and the isolated tree stump edges the rings get counted.
    Visualization of each step is saved under ./doc/ .
    """
    im = cv2.resize(cv2.imread(INPUT_FILE), (2000, 1500),
                    interpolation=cv2.INTER_LANCZOS4)

    filtered = cv2.medianBlur(im, 5)
    cv2.imwrite('./doc/median_filtered.png', filtered)
    denoised = cv2.fastNlMeansDenoisingColored(
        filtered, 
        h=NLMEANS_STRENGTH, 
        templateWindowSize=5, 
        searchWindowSize=15
    )
    cv2.imwrite('./doc/nlmeans_denoised.png', denoised)
    # ksize=(0,0) -> function chooses value for ksize according to sigma
    unsharp_mask = denoised - \
        cv2.GaussianBlur(denoised, ksize=(0, 0), sigmaX=2)
    sharpened = denoised + SHARPEN_STRENGTH * unsharp_mask
    cv2.imwrite('./doc/sharpened.png', sharpened)

    edges = cv2.Canny(sharpened, 75, 150)
    cv2.imwrite('./doc/canny_edges.png', edges)

    tree_mask = create_tree_mask(im)
    tree_edges = cv2.bitwise_and(edges, tree_mask)
    tree = cv2.bitwise_and(im, im, mask=tree_mask)

    center = approx_center(tree_edges)

    age = approx_age(tree_edges, center)
    print(f"Estimated age: {age} years")

    cv2.line(
        tree, 
        (0, center[1]), 
        (2000, center[1]), 
        color=255, 
        thickness=3
    )
    cv2.line(
        tree, 
        (center[0], 0), 
        (center[0], 1500), 
        color=255, 
        thickness=3
    )
    cv2.imwrite('./doc/original_with_mask_and_cross_section.png', tree)


if __name__ == "__main__":
    try:
        os.mkdir("doc")
    except:
        pass
    main()
