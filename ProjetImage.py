"""
import cv2
import numpy as np

image = cv2.imread('/users/licence/in07091/Images/piece.jpg', cv2.IMREAD_GRAYSCALE)

# Appliquer l'algorithme d'Otsu pour obtenir le seuil optimal
_, otsu_threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Afficher l'image seuillée avec le seuil personnalisé
cv2.imshow("image",otsu_threshold)

# Appliquer l'algorithme de Canny pour la détection de contours
edges = cv2.Canny(image, 100, 200)

# Trouver les contours dans l'image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Créer un masque vide de la même taille que l'image
mask = np.zeros_like(image)

# Créer des masques binaires pour chaque contour
for contour in contours:
    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
cv2.imshow("masque",mask)
# Appliquer les masques sur l'image originale pour extraire chaque pièce
pieces = cv2.bitwise_and(image, image, mask=mask)


cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
import cv2
import numpy as np

image = cv2.imread('/users/licence/in07091/Images/piece.jpg', cv2.IMREAD_GRAYSCALE)

# Appliquer l'algorithme d'Otsu pour obtenir le seuil optimal
_, otsu_threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Appliquer l'algorithme de Canny pour la détection de contours
edges = cv2.Canny(image, 100, 200)

# Trouver les contours dans l'image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Créer un masque vide de la même taille que l'image
mask = np.zeros_like(image)
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
pieces = cv2.bitwise_and(image, image, mask=mask)
print(nbPiece)
# Afficher l'image résultante
cv2.imshow("Image finale", pieces)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""

import cv2
import numpy as np

image = cv2.imread('/users/licence/in07091/Images/piece.jpg', cv2.IMREAD_GRAYSCALE)

# Appliquer l'algorithme d'Otsu pour obtenir le seuil optimal
_, otsu_threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Appliquer l'algorithme de Canny pour la détection de contours
edges = cv2.Canny(image, 100, 200)

# Trouver les contours dans l'image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialiser une liste pour stocker les contours significativement grands
big_contours = []

# Filtrer les contours en fonction de leur aire
for contour in contours:
    area = cv2.contourArea(contour)
    # Vous pouvez ajuster cette valeur de seuil en fonction de la taille de vos pièces
    if area > 300:  # Seuil empirique pour filtrer les petits contours
        big_contours.append(contour)

# Créer un masque vide de la même taille que l'image
mask = np.zeros_like(image)

# Dessiner les contours significativement grands sur le masque
cv2.drawContours(mask, big_contours, -1, (255), thickness=cv2.FILLED)

# Appliquer le masque sur l'image originale pour extraire les pièces
pieces = cv2.bitwise_and(image, image, mask=mask)

# Afficher le nombre de pièces détectées
print("Nombre de pièces détectées :", len(big_contours))

# Afficher l'image résultante
cv2.imshow("Image finale", pieces)
cv2.waitKey(0)
cv2.destroyAllWindows()
