import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

DIGITS_LOOKUP = {
    (1, 1, 1, 1, 1, 1, 0): 0,
    (1, 1, 0, 0, 0, 0, 0): 1,
    (1, 0, 1, 1, 0, 1, 1): 2,
    (1, 1, 1, 0, 0, 1, 1): 3,
    (1, 1, 0, 0, 1, 0, 1): 4,
    (0, 1, 1, 0, 1, 1, 1): 5,
    (0, 1, 1, 1, 1, 1, 1): 6,
    (1, 1, 0, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 0, 1, 1, 1): 9,
    (0, 0, 0, 0, 0, 1, 1): '-'
}
H_W_Ratio = 1.8
THRESHOLD = 50
arc_tan_theta = 6.0  # 数码管倾斜角度



def preprocess(img, threshold, show=False, kernel_size=(5, 5)):
    # 拉伸对比度
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    # 直方图局部均衡化
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(6, 6))
    img = clahe.apply(img)
    # 自适应阈值二值化
    dst = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, threshold)
    # 闭运算开运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)
    if show:
        cv2.imshow('equlizeHist', img)
        cv2.imshow('threshold', dst)
    return dst


def helper_extract(one_d_array, threshold=20):
    res = []
    flag = 0
    temp = 0
    for i in range(len(one_d_array)):
        if one_d_array[i] < 12 * 255:
            if flag > threshold:
                start = i - flag
                end = i
                temp = end
                if end - start > 20:
                    res.append((start, end))
            flag = 0
        else:
            flag += 1

    else:
        if flag > threshold:
            start = temp
            end = len(one_d_array)
            if end - start > 50:
                res.append((start, end))
    return res


def find_digits_positions(img, H_threshold=20, V_threshold=20, min_w=8, min_h=16, max_w=200, max_h=200):
    digits_positions = []
    horizontal_sum = np.sum(img, axis=0)
    horizontal_regions = helper_extract(horizontal_sum, threshold=H_threshold)
    vertical_sum = np.sum(img, axis=1)
    vertical_regions = helper_extract(vertical_sum, threshold=V_threshold)
    if len(vertical_regions) > 1:
        vertical_regions = [(vertical_regions[0][0], vertical_regions[-1][1])]
    for h_region in horizontal_regions:
        for v_region in vertical_regions:
            x0, x1 = h_region
            y0, y1 = v_region
            w = x1 - x0
            h = y1 - y0
            if w >= min_w and h >= min_h and w <= max_w and h <= max_h:
                digits_positions.append([(x0, y0), (x1, y1)])
    # 若找不到，尝试放宽阈值
    if len(digits_positions) == 0:
        for h_region in horizontal_regions:
            for v_region in vertical_regions:
                x0, x1 = h_region
                y0, y1 = v_region
                w = x1 - x0
                h = y1 - y0
                if w > 2 and h > 4:
                    digits_positions.append([(x0, y0), (x1, y1)])
    assert len(digits_positions) > 0, "Failed to find digits' positions using continuous algorithm."
    return digits_positions


def recognize_digits_area_method(digits_positions, output_img, input_img):
    digits = []
    for c in digits_positions:
        x0, y0 = c[0]
        x1, y1 = c[1]
        roi = input_img[y0:y1, x0:x1]
        h, w = roi.shape
        suppose_W = max(1, int(h / H_W_Ratio))

        # 对1的情况单独识别
        
        if w < suppose_W / 3:
            x0 = max(x0 + w - suppose_W, 0)
            roi = input_img[y0:y1, x0:x1]
            w = roi.shape[1]
        
        width = (max(int(w * 0.15), 1) + max(int(h * 0.15), 1)) // 2
        dhc = int(width * 0.8)
        small_delta = int(h / arc_tan_theta) // 4
        
        
        # Define segments for recognition
        segments = [
            ((w - width - small_delta, width // 2), (w, (h - dhc) // 2)),
            ((w - width - 2 * small_delta, (h + dhc) // 2), (w - small_delta, h - width // 2)),
            ((width - small_delta, h - width), (w - width - small_delta, h)),
            ((0, (h + dhc) // 2), (width, h - width // 2)),
            ((small_delta, width // 2), (small_delta + width, (h - dhc) // 2)),
            ((small_delta, 0), (w + small_delta, width)),
            ((width - small_delta, (h - dhc) // 2), (w - width - small_delta, (h + dhc) // 2))
        ]

        on = [0] * len(segments)

        # Visualize the segments for debugging
        for (i, ((xa, ya), (xb, yb))) in enumerate(segments):
            seg_roi = roi[ya:yb, xa:xb]
            cv2.rectangle(output_img[y0:y1, x0:x1], (xa, ya), (xb, yb), (128, 0, 0), 1)  # Draw segment rectangles
            total = cv2.countNonZero(seg_roi)
            area = (xb - xa) * (yb - ya) * 0.9
            if total / float(area) > 0.45:
                on[i] = 1

        # Recognize the digit based on active segments
        if tuple(on) in DIGITS_LOOKUP.keys():
            digit = DIGITS_LOOKUP[tuple(on)]
        else:
            digit = '*'

        digits.append(digit)

        # Draw bounding box and recognized digit on the output image
        cv2.rectangle(output_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(output_img, str(digit), (x0 + 3, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    return digits


def recognize_digits_line_method(digits_positions, output_img, input_img):
    digits = []
    for c in digits_positions:
        x0, y0 = c[0]
        x1, y1 = c[1]
        roi = input_img[y0:y1, x0:x1]
        h, w = roi.shape
        suppose_W = max(1, int(h / H_W_Ratio))

        # 消除无关符号干扰
        if x1 - x0 < 25 and cv2.countNonZero(roi) / ((y1 - y0) * (x1 - x0)) < 0.2:
            continue

        # 对1的情况单独识别
        
        if w < suppose_W / 1.5:
            x0 = max(x0 + int((w - suppose_W) * 0.8), 0)
            roi = input_img[y0:y1, x0:x1]
            w = roi.shape[1]
        
            
        center_y = h // 2
        quater_y_1 = h // 4
        quater_y_3 = quater_y_1 * 3
        center_x = w // 2
        line_width = 5  # line's width
        width = (max(int(w * 0.15), 1) + max(int(h * 0.15), 1)) // 2
        small_delta = int(h / arc_tan_theta) // 4
        segments = [
            ((w - 2 * width, quater_y_1 - line_width), (w, quater_y_1 + line_width)),
            ((w - 2 * width, quater_y_3 - line_width), (w, quater_y_3 + line_width)),
            ((center_x - line_width - small_delta, h - 2 * width), (center_x - small_delta + line_width, h)),
            ((0, quater_y_3 - line_width), (2 * width, quater_y_3 + line_width)),
            ((0, quater_y_1 - line_width), (2 * width, quater_y_1 + line_width)),
            ((center_x - line_width, 0), (center_x + line_width, 2 * width)),
            ((center_x - line_width, center_y - line_width), (center_x + line_width, center_y + line_width)),
        ]
        on = [0] * len(segments)

        # Visualize the segments for debugging
        for (i, ((xa, ya), (xb, yb))) in enumerate(segments):
            seg_roi = roi[ya:yb, xa:xb]
            cv2.rectangle(output_img[y0:y1, x0:x1], (xa, ya), (xb, yb), (128, 0, 0), 1)  # Draw segment rectangles
            total = cv2.countNonZero(seg_roi)
            area = (xb - xa) * (yb - ya) * 0.9
            if total / float(area) > 0.25:
                on[i] = 1

        if tuple(on) in DIGITS_LOOKUP.keys():
            digit = DIGITS_LOOKUP[tuple(on)]
        else:
            digit = '*'

        digits.append(digit)

        # 小数点的识别
        '''
        if cv2.countNonZero(roi[h - int(3 * width / 4):h, w - int(3 * width / 4):w]) / (9. / 16 * width * width) > 0.65:
            digits.append('.')
            cv2.rectangle(output_img,
                          (x0 + w - int(3 * width / 4), y0 + h - int(3 * width / 4)),
                          (x1, y1), (0, 128, 0), 2)
            cv2.putText(output_img, 'dot',
                        (x0 + w - int(3 * width / 4), y0 + h - int(3 * width / 4) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 128, 0), 2)
        '''
                        
        # Draw bounding box and recognized digit on the output image
        cv2.rectangle(output_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(output_img, str(digit), (x0 + 3, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    return digits
