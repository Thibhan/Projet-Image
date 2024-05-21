import cv2
import numpy as np

def resize_for_display(image, max_width=800, max_height=600):
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        return cv2.resize(image, new_size)
    return image

# Chargement de l'image en niveaux de gris
image = cv2.imread('C:\\Users\\mmuzz\\Desktop\\Projet Image\\Images\\.jpeg', cv2.IMREAD_GRAYSCALE)

# Vérification du chargement de l'image
if image is None:
    print("Erreur lors du chargement de l'image.")
    exit()

# Définition du pourcentage de redimensionnement
pourcentage = 50

# Calcul de la nouvelle taille en pourcentage
largeur = int(image.shape[1] * pourcentage / 100)
hauteur = int(image.shape[0] * pourcentage / 100)
nouvelle_taille = (largeur, hauteur)

# Redimensionnement de l'image
image_redimensionnee = cv2.resize(image, nouvelle_taille)
cv2.imshow("Image redimensionnée", resize_for_display(image_redimensionnee))

# Application d'un filtre gaussien
image_filtree = cv2.GaussianBlur(image_redimensionnee, (5, 5), 0)
cv2.imshow("Image filtrée", resize_for_display(image_filtree))

# Application de la méthode d'Otsu
_, otsu_threshold = cv2.threshold(image_filtree, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("Image après seuillage d'Otsu", resize_for_display(otsu_threshold))

# Détection des bords avec Canny
img_canny = cv2.Canny(image_filtree, 100, 200)
cv2.imshow("Contours détectés par Canny", resize_for_display(img_canny))

# Détection des cercles avec HoughCircles
circles = cv2.HoughCircles(
    img_canny,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=20,
    param1=100,
    param2=30,
    minRadius=10,
    maxRadius=100
)

# Création d'un masque vide de la même taille que l'image redimensionnée
mask = np.zeros_like(image_redimensionnee)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        center = (circle[0], circle[1])
        radius = circle[2]
        cv2.circle(mask, center, radius, (255), thickness=cv2.FILLED)

# Application des cercles du masque sur l'image originale
pieces = cv2.bitwise_and(image_redimensionnee, image_redimensionnee, mask=mask)
cv2.imshow("Masque appliqué", resize_for_display(pieces))

# Afficher le nombre de pièces détectées
nbPieces = len(circles[0, :]) if circles is not None else 0
print(f"Nombre de pièces détectées : {nbPieces}")

# Dessiner les cercles détectés
image_redimensionnee_copy = image_redimensionnee.copy()
if circles is not None:
    for circle in circles[0, :]:
        center = (circle[0], circle[1])
        radius = circle[2]
        cv2.circle(image_redimensionnee_copy, center, radius, (255), thickness=2)

# Affichage des cercles détectés
cv2.imshow("Cercles détectés", resize_for_display(image_redimensionnee_copy))
cv2.waitKey(0)
cv2.destroyAllWindows()