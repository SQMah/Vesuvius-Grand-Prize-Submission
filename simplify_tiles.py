import cv2
import numpy as np
import matplotlib.pyplot as plt


def process_image(image_path, threshold_percent):
    # Load the image in grayscale (0 - black, 255 - white)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    print(img.dtype)
    out = np.zeros(img.shape, dtype=np.uint8)

    # Window size and stride
    window_size = 64
    stride = 64

    # Iterate over the image
    for y in range(0, img.shape[0], stride):
        for x in range(0, img.shape[1], stride):
            window = img[y:y + window_size, x:x + window_size]

            # Check if the window is complete
            if window.shape[0] == window_size and window.shape[1] == window_size:
                # Calculate the percentage of white pixels
                white_pixels = np.sum(window >= 100)
                total_pixels = window_size * window_size
                white_percent = (white_pixels / total_pixels) * 100

                # If the percentage of white exceeds the threshold, set the window to white
                if white_percent >= threshold_percent:
                    out[y:y + window_size, x:x + window_size] = 255

    return out


if __name__ == '__main__':
    # Replace 'path_to_your_image.jpg' with your image path
    processed_img = process_image('data/scroll1_hari/20230827161847/layers/30.tif',
                                  threshold_percent=50)  # 50% threshold

    cv2.imshow("processed image", processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
