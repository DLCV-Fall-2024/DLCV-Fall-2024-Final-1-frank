import cv2


def simplify_contour(contour, n_corners=4):
    # https://stackoverflow.com/questions/13028961/how-to-force-approxpolydp-to-return-only-the-best-4-corners-opencv-2-4-2
    n_iter, max_iter = 0, 1000
    lb, ub = 0., 1.

    while True:
        n_iter += 1
        if n_iter > max_iter:
            # return contour
            return approx

        k = (lb + ub) / 2.
        eps = k * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)

        if len(approx) > n_corners:
            lb = (lb + ub) / 2.
        elif len(approx) < n_corners:
            ub = (lb + ub) / 2.
        else:
            return approx
