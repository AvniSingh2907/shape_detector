import cv2
import numpy as np

# Generate Shapes
image = np.ones((500, 500, 3), dtype=np.uint8) * 255

pts_triangle = np.array([[100, 100], [50, 200], [150, 200]], np.int32)
cv2.fillPoly(image, [pts_triangle], (0, 150, 0))

cv2.rectangle(image, (250, 80), (450, 160), (0, 0, 200), -1)
cv2.rectangle(image, (70, 300), (170, 400), (200, 0, 0), -1)

pts_hexagon = np.array([[350, 300], [420, 320], [420, 380], [350, 400], [280, 380], [280, 320]], np.int32)
cv2.fillPoly(image, [pts_hexagon], (150, 0, 150))

# Preprocessing
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
edges = cv2.Canny(thresholded, 50, 150)

# Contour Detection & Filtering
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 500:
        continue
        
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)

    # Approximation & Classification
    epsilon = 0.04 * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    vertices = len(approx)
    shape_name = "Polygon"

    if vertices == 3:
        shape_name = "Triangle"
    elif vertices == 4:
        aspect_ratio = float(w) / h
        shape_name = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
    elif vertices == 5:
        shape_name = "Pentagon"
    elif vertices == 6:
        shape_name = "Hexagon"
    elif vertices > 6:
        shape_name = "Circle/Polygon"

    # Visualization Outputs
    cv2.drawContours(image, [approx], 0, (0, 255, 0), 3)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
    
    label = f"{shape_name} | Area:{int(area)} | Peri:{int(perimeter)}"
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

# Windows Display
cv2.imshow("1A & 1B: Blurred Grayscale", blurred)
cv2.imshow("1C: Thresholded Binary", thresholded)
cv2.imshow("1D: Canny Edges", edges)
cv2.imshow("Final Assignment Output", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
