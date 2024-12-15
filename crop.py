import cv2
import numpy as np
import os
from extractRect import *

INF = 1e9


def mindiff(a, b):
    a = list(a)
    b = list(b)
    if a[1] == b[1]:
        return INF
    slope = (b[0] - a[0]) / (b[1] - a[1])
    return abs(slope)


def rectSlope(cds):
    if len(cds) != 4:
        return INF

    tA, tB, tC, tD = sorted([list(c[0]) for c in cds])
    return mindiff(tA, tB) + mindiff(tC, tD) + mindiff(reversed(tA), reversed(tC)) + mindiff(reversed(tB), reversed(tD))


def remainred(img):
    # https://stackoverflow.com/questions/30331944/finding-red-color-in-image-using-python-opencv
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower mask (0-10) -> (0-50)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([50, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180) -> (160-190)
    lower_red = np.array([160, 50, 50])
    upper_red = np.array([190, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0 + mask1

    # set my output img to zero everywhere except my mask
    output_img = img.copy()
    output_img[np.where(mask == 0)] = 0

    return output_img


pth = "./cropped_example"
for idx in ["2", "3", "Train_regional_1044", "Train_regional_1107", "Train_regional_1230", "Train_regional_1231"]:
    image = cv2.imread(os.path.join(pth, f"{idx}.jpg"))
    if image is None:
        raise IOError("Could not load the image.")

    image_cp = remainred(image)
    image_cp[:, :, 0] = 255
    image_cp[:, :, 1] = 255
    # image_rgb[:, :, 2] = 255 - image_rgb[:, :, 2]
    gray = cv2.cvtColor(image_cp, cv2.COLOR_BGR2GRAY)

    # Smooth the image to avoid noise, which is similar to converting the image to jpg and back to png
    # https://stackoverflow.com/questions/58466190/how-to-smooth-lines-in-the-image
    # kernel = np.ones((15, 15), np.uint8)
    # gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # cv2.imwrite(f"{idx}_gray.png", gray)

    # Threshold to isolate bright (white) areas
    thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 0)
    _, thresh2 = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_and(thresh1, thresh2)

    # thresh = 255 - thresh
    # cv2.imwrite(f"{idx}_thresh1.png", thresh1)
    # cv2.imwrite(f"{idx}_thresh2.png", thresh2)
    cv2.imwrite(os.path.join(pth, f"{idx}_thresh.png"), thresh)

    # Find contours of the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # blank = np.zeros_like(image)
    # cv2.drawContours(blank, cts, -1, (255, 255, 255), -1)
    # cv2.imwrite(f"{idx}_contours.png", blank)

    if not contours:
        print(f"{idx} - No contours found.")
        continue

    blank = np.zeros_like(image)
    cv2.drawContours(blank, contours, -1, (255, 255, 255), -1)
    cv2.imwrite(os.path.join(pth, f"{idx}_contour.png"), blank)

    cts = []
    for ct in reversed(sorted(contours, key=cv2.contourArea)):
        if cv2.contourArea(ct) < 10:
            break
        # https://stackoverflow.com/questions/47520487/how-to-use-python-opencv-to-find-largest-connected-component-in-a-single-channel
        # labels, stats = cv2.connectedComponentsWithStats(thresh, connectivity=8)[1:3]
        # largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        # blank[labels == largest_label] = 255

        # approx = cv2.convexHull(approx, returnPoints = True)
        # ct = simplify_contour(ct, 4)
        # outer_polygon = np.array([list(c[0]) for c in ct])

        blank = np.zeros_like(image)
        cv2.drawContours(blank, [ct], -1, (255, 255, 255), -1)
        blank = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)[::-1].T

        # change with 0 and 1
        idx_in = np.where(blank == 255)
        # idx_out = np.where(blank == 0)
        blk = np.ones_like(blank)
        blk[idx_in] = 0

        # https://github.com/pogam/ExtractRect
        # https://stackoverflow.com/questions/32674256/how-to-adapt-or-resize-a-rectangle-inside-an-object-without-including-or-with-a
        try:
            rect_coord_ori, angle, coord_out_rot = findRotMaxRect(
                blk,
                flag_opt=True,
                nbre_angle=4,
                flag_parallel=False,
                flag_out='rotation',
                flag_enlarge_img=False,
                # limit_image_size=100
            )

            ct = np.int0([[[x, image.shape[0] - y]] for x, y in rect_coord_ori])
            slope = rectSlope(ct)
            if slope < 0.1:
                cts.append(ct)
        except Exception as e:
            break

    approx = max(cts, key=cv2.contourArea)

    # [x, y], [a, b], angle = cv2.fitEllipse(ct)
    # angle = np.radians(angle)
    # init = [[-a / 2, b / 2], [a / 2, b / 2], [a / 2, -b / 2], [-a / 2, -b / 2]]
    # rot_mat = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    # ct = np.int0([[[x, y] + np.dot(rot_mat, pt)] for pt in init])

    # https: // stackoverflow.com / questions / 61166180 / detect - rectangles - in -opencv - 4 - 2 - 0 - using - python - 3 - 7
    # best_contour = max(contours, key=cv2.contourArea)
    # Approximate the contour to a polygon and ensure it is somewhat rectangular

    # Get the minimum bounding rectangle's (center (x,y), (width, height), rotation angle).
    # app = cv2.minAreaRect(np.array([list(c[0]) for c in approx]))
    # approx = np.int0([[it] for it in cv2.boxPoints(app)])

    # cv2.drawContours(image, li, -1, (255, 255, 255), -1)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [approx], (255, 255, 255))
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask_binary = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)

    # Use the mask to isolate the rectangle from the original image
    result = np.zeros_like(image)
    result[mask_binary == 255] = image[mask_binary == 255]
    cv2.imwrite(os.path.join(pth, f"{idx}_output.png"), result)
