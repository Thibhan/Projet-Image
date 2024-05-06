import cv2
import numpy as np

image = cv2.imread('/users/licence/in07091/Images/piece_img.jpg', cv2.IMREAD_GRAYSCALE)
pourcentage = 50

# Calculer la nouvelle taille en pourcentage
largeur = int(image.shape[1] * pourcentage / 100)
hauteur = int(image.shape[0] * pourcentage / 100)
nouvelle_taille = (largeur, hauteur)

# Redimensionner l'image en utilisant la méthode cv2.resize
image_redimensionnee = cv2.resize(image, nouvelle_taille)
_, otsu_threshold = cv2.threshold(image_redimensionnee, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow("image", otsu_threshold)

img_canny = cv2.Canny(image_redimensionnee, 100, 200)

contours, _ = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

circles = []

for contour in contours:
    area = cv2.contourArea(contour)
    if area > 300:  
        # Approximation du contour pour avoir un polygone
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Si le polygone est presque circulaire, considérez-le comme un cercle
        if len(approx) >= 4:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            circles.append((center, radius))

# Supprimer les cercles imbriqués en conservant uniquement le cercle le plus extérieur
if len(circles) > 1:
    circles.sort(key=lambda x: x[1], reverse=True)  # Tri par rayon décroissant
    circles_copy = circles.copy()
    for circle in circles_copy:
        for other_circle in circles_copy:
            if circle != other_circle:
                distance = np.linalg.norm(np.array(circle[0]) - np.array(other_circle[0]))
                if distance < circle[1] - other_circle[1]:
                    circles.remove(other_circle)

# Dessiner les cercles détectés
for circle in circles:
    cv2.circle(image_redimensionnee, circle[0], circle[1], (255), thickness=2)

cv2.imshow("Cercles détectés", image_redimensionnee)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
masque = np.zeros_like(image_redimensionnee)

for contour in circles:
    cv2.drawContours(masque, [contour], -1, (255), thickness=cv2.FILLED)
cv2.imshow("masque", masque)
pieces = cv2.bitwise_and(image_redimensionnee, image_redimensionnee, mask=masque)

cv2.imshow("pieces", pieces)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
import cv2
import numpy as np

image = cv2.imread('/users/licence/in07091/Images/piece_img.jpg', cv2.IMREAD_GRAYSCALE)
pourcentage = 50

# Calculer la nouvelle taille en pourcentage
largeur = int(image.shape[1] * pourcentage / 100)
hauteur = int(image.shape[0] * pourcentage / 100)
nouvelle_taille = (largeur, hauteur)

# Redimensionner l'image en utilisant la méthode cv2.resize
image_redimensionnee = cv2.resize(image, nouvelle_taille)
_, otsu_threshold = cv2.threshold(image_redimensionnee, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow("image", otsu_threshold)

img_canny = cv2.Canny(image_redimensionnee, 100, 200)

contours, _ = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

circles = []

for contour in contours:
    area = cv2.contourArea(contour)
    if area > 290:  
        # Approximation du contour pour avoir un polygone
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Si le polygone est presque circulaire, considérez-le comme un cercle
        if len(approx) >= 6:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            circles.append((center, radius))

# Supprimer les cercles imbriqués en conservant uniquement le cercle le plus extérieur
if len(circles) > 1:
    circles.sort(key=lambda x: x[1], reverse=True)  # Tri par rayon décroissant
    circles_copy = circles.copy()
    for circle in circles_copy:
        for other_circle in circles_copy:
            if circle != other_circle:
                distance = np.linalg.norm(np.array(circle[0]) - np.array(other_circle[0]))
                if distance < circle[1] - other_circle[1]:
                    circles.remove(other_circle)

# Dessiner les cercles détectés
for circle in circles:
    cv2.circle(image_redimensionnee, circle[0], circle[1], (255), thickness=2)

cv2.imshow("Cercles détectés", image_redimensionnee)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
