import cv2


def high_pass_filter(img):
    blur = cv2.GaussianBlur(img, (5,5), 100)
    diff = cv2.subtract(img, blur)
    sharp = cv2.add(img, diff)
    return sharp

def sobel(img):
    
    grad_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=1)
    grad_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=1)
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    uns_grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    #smoothing sobel    
    smooth_grad = cv2.GaussianBlur(uns_grad, (5,5), 0)
 
    smooth_sobel = cv2.add(img, smooth_grad)

    sobel = cv2.add(img, uns_grad)
    return sobel, smooth_sobel
        
