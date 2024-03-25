import cv2
import numpy as np

# Charger l'image
image = cv2.imread('votre_image.jpg', cv2.IMREAD_GRAYSCALE)

# Appliquer l'algorithme de Canny pour la détection de contours
edges = cv2.Canny(image, 100, 200)

# Trouver les contours dans l'image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Créer un masque vide de la même taille que l'image
mask = np.zeros_like(image)

# Créer des masques binaires pour chaque contour
for contour in contours:
    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

# Appliquer les masques sur l'image originale pour extraire chaque pièce
pieces = cv2.bitwise_and(image, image, mask=mask)

# Afficher ou enregistrer les pièces
for i, piece in enumerate(pieces):
    cv2.imshow('Piece {}'.format(i), piece)
    cv2.imwrite('piece_{}.jpg'.format(i), piece)

cv2.waitKey(0)
cv2.destroyAllWindows()
