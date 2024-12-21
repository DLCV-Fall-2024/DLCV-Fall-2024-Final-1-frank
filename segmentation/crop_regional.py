from __future__ import print_function
from __future__ import division
import cv2
import numpy as np
import os
from PIL import Image
from builtins import range
from past.utils import old_div
from scipy import optimize
import pdb
import multiprocessing
from functools import reduce
import json
from tqdm import tqdm

INF = 1e9


def findMaxRect(data):
    '''http://stackoverflow.com/a/30418912/5008845'''

    nrows, ncols = data.shape
    w = np.zeros(dtype=int, shape=data.shape)
    h = np.zeros(dtype=int, shape=data.shape)
    skip = 1
    area_max = (0, [])

    for r in range(nrows):
        for c in range(ncols):
            if data[r][c] == skip:
                continue
            if r == 0:
                h[r][c] = 1
            else:
                h[r][c] = h[r - 1][c] + 1
            if c == 0:
                w[r][c] = 1
            else:
                w[r][c] = w[r][c - 1] + 1
            minw = w[r][c]
            for dh in range(h[r][c]):
                minw = min(minw, w[r - dh][c])
                area = (dh + 1) * minw
                if area > area_max[0]:
                    area_max = (area, [(r - dh, c - minw + 1, r, c)])

    return area_max


########################################################################
def residual(angle, data):

    nx, ny = data.shape
    M = cv2.getRotationMatrix2D((old_div((nx - 1), 2), old_div((ny - 1), 2)), int(angle), 1)
    RotData = cv2.warpAffine(data, M, (nx, ny), flags=cv2.INTER_NEAREST, borderValue=1)
    rectangle = findMaxRect(RotData)

    return 1. / rectangle[0]


########################################################################
def residual_star(args):
    return residual(*args)


########################################################################
def get_rectangle_coord(angle, data, flag_out=None):
    nx, ny = data.shape
    M = cv2.getRotationMatrix2D((old_div((nx - 1), 2), old_div((ny - 1), 2)), angle, 1)
    RotData = cv2.warpAffine(data, M, (nx, ny), flags=cv2.INTER_NEAREST, borderValue=1)
    rectangle = findMaxRect(RotData)

    if flag_out:
        return rectangle[1][0], M, RotData
    else:
        return rectangle[1][0], M


########################################################################
def findRotMaxRect(data_in, flag_opt=False, flag_parallel=False, nbre_angle=10, flag_out=None, flag_enlarge_img=False, limit_image_size=300):
    '''
    flag_opt     : True only nbre_angle are tested between 90 and 180 
                        and a opt descent algo is run on the best fit
                   False 100 angle are tested from 90 to 180.
    flag_parallel: only valid when flag_opt=False. the 100 angle are run on multithreading
    flag_out     : angle and rectangle of the rotated image are output together with the rectangle of the original image
    flag_enlarge_img : the image used in the function is double of the size of the original to ensure all feature stay in when rotated
    limit_image_size : control the size numbre of pixel of the image use in the function. 
                       this speeds up the code but can give approximated results if the shape is not simple
    '''

    # time_s = datetime.datetime.now()

    # make the image square
    # ----------------
    nx_in, ny_in = data_in.shape
    if nx_in != ny_in:
        n = max([nx_in, ny_in])
        data_square = np.ones([n, n])
        xshift = old_div((n - nx_in), 2)
        yshift = old_div((n - ny_in), 2)
        if yshift == 0:
            data_square[xshift:(xshift + nx_in), :] = data_in[:, :]
        else:
            data_square[:, yshift:(yshift + ny_in)] = data_in[:, :]
    else:
        xshift = 0
        yshift = 0
        data_square = data_in

    # apply scale factor if image bigger than limit_image_size
    # ----------------
    if data_square.shape[0] > limit_image_size:
        data_small = cv2.resize(data_square, (limit_image_size, limit_image_size), interpolation=0)
        scale_factor = old_div(1. * data_square.shape[0], data_small.shape[0])
    else:
        data_small = data_square
        scale_factor = 1

    # set the input data with an odd number of point in each dimension to make rotation easier
    # ----------------
    nx, ny = data_small.shape
    nx_extra = -nx
    ny_extra = -ny
    if nx % 2 == 0:
        nx += 1
        nx_extra = 1
    if ny % 2 == 0:
        ny += 1
        ny_extra = 1
    data_odd = np.ones([data_small.shape[0] + max([0, nx_extra]), data_small.shape[1] + max([0, ny_extra])])
    data_odd[:-nx_extra, :-ny_extra] = data_small
    nx, ny = data_odd.shape

    nx_odd, ny_odd = data_odd.shape

    if flag_enlarge_img:
        data = np.zeros([2 * data_odd.shape[0] + 1, 2 * data_odd.shape[1] + 1]) + 1
        nx, ny = data.shape
        data[old_div(nx, 2) - old_div(nx_odd, 2):old_div(nx, 2) + old_div(nx_odd, 2), old_div(ny, 2) - old_div(ny_odd, 2):old_div(ny, 2) + old_div(ny_odd, 2)] = data_odd
    else:
        data = np.copy(data_odd)
        nx, ny = data.shape

    # print (datetime.datetime.now()-time_s).total_seconds()

    if flag_opt:
        myranges_brute = ([(90., 180.),])
        coeff0 = np.array([0.,])
        coeff1 = optimize.brute(residual, myranges_brute, args=(data,), Ns=nbre_angle, finish=None)
        popt = optimize.fmin(residual, coeff1, args=(data,), xtol=5, ftol=1.e-5, disp=False)
        angle_selected = popt[0]

        # rotation_angle = np.linspace(0,360,100+1)[:-1]
        # mm = [residual(aa,data) for aa in rotation_angle]
        # plt.plot(rotation_angle,mm)
        # plt.show()
        # pdb.set_trace()

    else:
        rotation_angle = np.linspace(90, 180, 100 + 1)[:-1]
        args_here = []
        for angle in rotation_angle:
            args_here.append([angle, data])

        if flag_parallel:

            # set up a pool to run the parallel processing
            cpus = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(processes=cpus)

            # then the map method of pool actually does the parallelisation

            results = pool.map(residual_star, args_here)

            pool.close()
            pool.join()

        else:
            results = []
            for arg in args_here:
                results.append(residual_star(arg))

        argmin = np.array(results).argmin()
        angle_selected = args_here[argmin][0]
    rectangle, M_rect_max, RotData = get_rectangle_coord(angle_selected, data, flag_out=True)
    # rectangle, M_rect_max  = get_rectangle_coord(angle_selected,data)

    # print (datetime.datetime.now()-time_s).total_seconds()

    # invert rectangle
    M_invert = cv2.invertAffineTransform(M_rect_max)
    rect_coord = [rectangle[:2], [rectangle[0], rectangle[3]],
                  rectangle[2:], [rectangle[2], rectangle[1]]]

    # ax = plt.subplot(111)
    # ax.imshow(RotData.T,origin='lower',interpolation='nearest')
    # patch = patches.Polygon(rect_coord, edgecolor='k', facecolor='None', linewidth=2)
    # ax.add_patch(patch)
    # plt.show()

    rect_coord_ori = []
    for coord in rect_coord:
        rect_coord_ori.append(np.dot(M_invert, [coord[0], (ny - 1) - coord[1], 1]))

    # transform to numpy coord of input image
    coord_out = []
    for coord in rect_coord_ori:
        coord_out.append([scale_factor * round(coord[0] - (old_div(nx, 2) - old_div(nx_odd, 2)), 0) - xshift,
                          scale_factor * round((ny - 1) - coord[1] - (old_div(ny, 2) - old_div(ny_odd, 2)), 0) - yshift])

    coord_out_rot = []
    coord_out_rot_h = []
    for coord in rect_coord:
        coord_out_rot.append([scale_factor * round(coord[0] - (old_div(nx, 2) - old_div(nx_odd, 2)), 0) - xshift,
                              scale_factor * round(coord[1] - (old_div(ny, 2) - old_div(ny_odd, 2)), 0) - yshift])
        coord_out_rot_h.append([scale_factor * round(coord[0] - (old_div(nx, 2) - old_div(nx_odd, 2)), 0),
                                scale_factor * round(coord[1] - (old_div(ny, 2) - old_div(ny_odd, 2)), 0)])

    # M = cv2.getRotationMatrix2D( ( (data_square.shape[0]-1)/2, (data_square.shape[1]-1)/2 ), angle_selected,1)
    # RotData = cv2.warpAffine(data_square,M,data_square.shape,flags=cv2.INTER_NEAREST,borderValue=1)
    # ax = plt.subplot(121)
    # ax.imshow(data_square.T,origin='lower',interpolation='nearest')
    # ax = plt.subplot(122)
    # ax.imshow(RotData.T,origin='lower',interpolation='nearest')
    # patch = patches.Polygon(coord_out_rot_h, edgecolor='k', facecolor='None', linewidth=2)
    # ax.add_patch(patch)
    # plt.show()

    # coord for data_in
    # ----------------
    # print scale_factor, xshift, yshift
    # coord_out2 = []
    # for coord in coord_out:
    #    coord_out2.append([int(np.round(scale_factor*coord[0]-xshift,0)),int(np.round(scale_factor*coord[1]-yshift,0))])

    # print (datetime.datetime.now()-time_s).total_seconds()

    if flag_out is None:
        return coord_out
    elif flag_out == 'rotation':
        return coord_out, angle_selected, coord_out_rot
    else:
        print('bad def in findRotMaxRect input. stop')
        pdb.set_trace()

######################################################


def factors(n):
    return set(reduce(list.__add__,
                      ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))


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

def crop(image_path):
    image = cv2.imread(image_path)
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
    # cv2.imwrite(os.path.join(pth, f"{idx}_thresh.png"), thresh)

    # Find contours of the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # blank = np.zeros_like(image)
    # cv2.drawContours(blank, cts, -1, (255, 255, 255), -1)
    # cv2.imwrite(f"{idx}_contours.png", blank)

    if not contours:
        print(f"No contours found.")
        return None

    blank = np.zeros_like(image)
    cv2.drawContours(blank, contours, -1, (255, 255, 255), -1)
    # cv2.imwrite(os.path.join(pth, f"{idx}_contour.png"), blank)

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
    if not cts:
        return None
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
    
    result = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    _, mask_binary = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)

    # find coordinates
    coords = cv2.findNonZero(mask_binary)  

    # find the min, max
    x_min, y_min, w, h = cv2.boundingRect(coords)
    x_max = x_min + w - 1
    y_max = y_min + h - 1

    return result, (x_min, y_min, x_max, y_max)
    
if __name__ == "__main__":
    OPEN_IMAGE_DIR = "data/train/images"
    SAVE_IMAGE_DIR = "data/regional_segmentation"
    os.makedirs(SAVE_IMAGE_DIR, exist_ok=True)
    
    regional_list = [ele for ele in os.listdir(OPEN_IMAGE_DIR) if "regional" in ele]
    
    coord_list = dict()
    for img_name in tqdm(regional_list):
        img_path = os.path.join(OPEN_IMAGE_DIR, img_name)
        try:
            img, coord = crop(image_path=img_path)
            img.save(os.path.join(SAVE_IMAGE_DIR, img_name))
            id = img_name.split('.')[0]
            coord_list[id] = coord
        except Exception as e:
            print(e)
    with open(os.path.join(SAVE_IMAGE_DIR, "regional_coord.json"), 'w') as f:
        json.dump(coord_list, f, indent=4)