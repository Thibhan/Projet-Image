import cv2
import numpy as np

image = cv2.imread('/users/licence/in07091/Images/piece_img3.jpg', cv2.IMREAD_GRAYSCALE)
pourcentage = 50

# Calculer la nouvelle taille en pourcentage
largeur = int(image.shape[1] * pourcentage / 100)
hauteur = int(image.shape[0] * pourcentage / 100)
nouvelle_taille = (largeur, hauteur)

# Redimensionner l'image en utilisant la méthode cv2.resize
image_redimensionnee = cv2.resize(image, nouvelle_taille)

# Appliquer un filtre gaussien
image_filtree = cv2.GaussianBlur(image_redimensionnee, (5, 5), 0)

# Appliquer la méthode d'Otsu
_, otsu_threshold = cv2.threshold(image_filtree, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow("image", otsu_threshold)

img_canny = cv2.Canny(image_filtree, 100, 200)

cv2.imshow("img_canny", img_canny)

contours, _ = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Créer un masque vide de la même taille que l'image redimensionnée
mask = np.zeros_like(image_redimensionnee)
nbPiece = 0
# Trouver les contours des cercles dans le masque
for contour in contours:
    # Approximer le contour pour un polygone
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
    
    # Si le polygone est un cercle (approximativement)
    if len(approx) > 8:
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
        nbPiece +=1

# Appliquer les cercles du masque sur l'image originale
pieces = cv2.bitwise_and(image_redimensionnee, image_redimensionnee, mask=mask)

# Trouver les contours des pièces
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
circles = []

for contour in contours:
    area = cv2.contourArea(contour)
    if area > 150:  
        # Approximation du contour pour avoir un polygone
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Si le polygone est presque circulaire, considérez-le comme un cercle
        if len(approx) >= 1:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            circles.append((center, radius))

"""
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

"""
# Dessiner les cercles détectés
image_redimensionnee_copy = image_redimensionnee.copy()
for circle in circles:
    cv2.circle(image_redimensionnee_copy, circle[0], circle[1], (255), thickness=2)

cv2.imshow("Cercles détectés", image_redimensionnee_copy)
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
