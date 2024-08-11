import cv2
import numpy as np

def angle_between(v1, v2):
    """Calculate the angle between two vectors."""
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return np.degrees(angle)

def has_straight_edges(approx, tolerance=10):
    """Check if all angles in the shape are close to 90 or 180 degrees."""
    for i in range(len(approx)):
        p1 = approx[i][0]
        p2 = approx[(i + 1) % len(approx)][0]
        p3 = approx[(i + 2) % len(approx)][0]
        
        v1 = p1 - p2
        v2 = p3 - p2
        angle = angle_between(v1, v2)
        
        if not (90 - tolerance <= angle <= 90 + tolerance or 180 - tolerance <= angle <= 180 + tolerance):
            return False
    return True

def find_non_overlapping_position(positions, x, y, w, h, image_width, image_height, margin=10):
    """Find a position for text that does not overlap with existing positions."""
    new_x, new_y = x, y
    for (px, py, pw, ph) in positions:
        if abs(new_x - px) < pw + margin and abs(new_y - py) < ph + margin:
            new_y = py + ph + margin
            
            if new_y + h > image_height:
                new_y = y
                new_x = px + pw + margin
                if new_x + w > image_width:
                    new_x = x
                    new_y = y + h + margin
    return new_x, new_y

def detect_shapes(image_path):

    
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not loaded. Check the file path.")

    
    image = cv2.resize(image, (1000, 1000))
    image_height, image_width = image.shape[:2]

    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 150)

    
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    positions = []  

    for i, contour in enumerate(contours):
       
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        
        x, y, w, h = cv2.boundingRect(approx)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        
        shape_name = "Unidentified"
        if len(approx) == 3:
            if has_straight_edges(approx):
                shape_name = "Triangle"
        elif len(approx) == 4:
            aspect_ratio = float(w) / h
            if 0.8 <= aspect_ratio <= 1.2 and has_straight_edges(approx):
                shape_name = "Rectangle/Square"
        elif len(approx) == 5 and has_straight_edges(approx):
            shape_name = "Pentagon"
        elif len(approx) == 6 and has_straight_edges(approx):
            shape_name = "Hexagon"
        elif 7 <= len(approx) <= 12 and has_straight_edges(approx):
            shape_name = "Star"
        else:
            area = cv2.contourArea(contour)
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            circularity = area / (np.pi * (radius ** 2))
            if 0.7 <= circularity <= 1.3:
                shape_name = "Circle/Ellipse"
        
        
        text_size = cv2.getTextSize(shape_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        new_x, new_y = find_non_overlapping_position(positions, x, y - 10, text_size[0], text_size[1], image_width, image_height)
        positions.append((new_x, new_y, text_size[0], text_size[1]))

        
        cv2.putText(image, shape_name, (new_x, new_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        
        if hierarchy[0][i][3] != -1:
            inner_shape_name = shape_name + " Inside"
            inner_text_size = cv2.getTextSize(inner_shape_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            inner_x, inner_y = find_non_overlapping_position(positions, x + w + 10, y, inner_text_size[0], inner_text_size[1], image_width, image_height)
            positions.append((inner_x, inner_y, inner_text_size[0], inner_text_size[1]))
            cv2.putText(image, inner_shape_name, (inner_x, inner_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    
    cv2.imshow('Detected Shapes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = 'curve.png' 
detect_shapes(image_path)


