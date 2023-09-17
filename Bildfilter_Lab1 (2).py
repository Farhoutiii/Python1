import cv2 
import numpy as np 

def apply_filters(img):

    images = {}
    
    # Ungefiltertes Bild
    images['Original'] = img
    
    # GaussianBlur
    images['GaussianBlur_3_3'] = cv2.GaussianBlur(img, (3,3), sigmaX=0)
    images['GaussianBlur_55_3'] = cv2.GaussianBlur(img, (55,3), sigmaX=0)
    
    # MedianBlur
    images['MedianBlur_3_3'] = cv2.medianBlur(img, 3)
    
    # Laplacian
    #1
    laplacian = cv2.Laplacian(img, cv2.CV_32F)
    #1
    laplacian_norm = cv2.normalize(laplacian, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    images['Laplacian'] = laplacian_norm
    
    # Sobel x-Richtung
    sobel_x = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    sobel_x_norm = cv2.normalize(sobel_x, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    images['Sobel_x'] = sobel_x_norm
    
    # Sobel y-Richtung
    sobel_y = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    sobel_y_norm = cv2.normalize(sobel_y, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    images['Sobel_y'] = sobel_y_norm
    
    # Canny
    edges = cv2.Canny(img, 50, 150)
    images['Canny'] = edges
    
    # Dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilate = cv2.dilate(img, kernel, anchor=(-1,-1), iterations=1)
    images['Dilate_7x7'] = dilate
    
    # Erode
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    erode = cv2.erode(img, kernel, anchor=(-1,-1), iterations=1)
    images['Erode_17x_17'] = erode
    
    # Binarization
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    images['Binarisierung_otsu'] = binarized
    
    

    # Histogram equalization
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    images['Histogram_equalization']= equalized

    #Show Imgs
    #Forschlife um alle images anzuzeigen
    for i in images:
        cv2.imshow(i, images[i])

    c = cv2.waitKey(10)
    #save imgs
    if(c== ord('2')):             
        for i in images:
            cv2.imwrite(f"{i}.png", images[i])

# Konfiguration der Videoquelle
vc = cv2.VideoCapture(0)
vc.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)



# Endlosschleife für die Bildverarbeitung
while True:
    # Bilder von der Kamera holen und skalieren
    ret, frame = vc.read()
    frame = cv2.resize(frame, (640, 480))
    
    apply_filters(frame)
    c = cv2.waitKey(10)
    
    if(c==ord('q')):
        break
    