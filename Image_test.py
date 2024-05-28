import cv2
import numpy as np
import os
import json

def resize_for_display(image, max_width=800, max_height=600):
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        return cv2.resize(image, new_size)
    return image

def process_image(image_path,image_annotations):

    filtered_circles = None

    image_clr = cv2.imread(image_path)
    # Chargement de l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Vérification du chargement de l'image
    if image is None:
        print(f"Erreur lors du chargement de l'image : {image_path}")
        return

    # Définition du pourcentage de redimensionnement
    pourcentage = 50

    # Calcul de la nouvelle taille en pourcentage
    largeur = int(image.shape[1] * pourcentage / 100)
    hauteur = int(image.shape[0] * pourcentage / 100)
    nouvelle_taille = (largeur, hauteur)

    # Redimensionnement de l'image
    image_redimensionnee = cv2.resize(image, nouvelle_taille)
    image_clr_redimensionnee = cv2.resize(image_clr, nouvelle_taille)

    # Application d'un filtre gaussien avec un noyau de taille plus grande
    image_filtree = cv2.GaussianBlur(image_redimensionnee, (5, 5), 0)
    #cv2.imshow("Image filtrée", resize_for_display(image_filtree))

    # Détection des bords avec Canny
    img_canny = cv2.Canny(image_filtree, 100, 200)
    #cv2.imshow("Contours détectés par Canny", resize_for_display(img_canny))

    image_canny_filtree = cv2.GaussianBlur(img_canny, (5, 5), 0)

    # Détection des cercles avec HoughCircles
    circles = cv2.HoughCircles(
        image_canny_filtree,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,  # Augmenter cette valeur pour réduire le nombre de cercles proches
        param1=110,
        param2=30,  # Ajuster cette valeur pour affiner la détection
        minRadius=30,  # Augmenter cette valeur pour ignorer les petits cercles
        maxRadius=300  # Réduire cette valeur pour ignorer les grands cercles
    )

    # Création d'un masque vide de la même taille que l'image redimensionnée
    mask = np.zeros_like(image_redimensionnee)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        filtered_circles = []

        for circle in circles[0, :]:
            keep = True
            for other_circle in filtered_circles:
                distance = np.linalg.norm(np.array(circle[:2]) - np.array(other_circle[:2]))
                if distance < (circle[2] + other_circle[2]):  # check if circles overlap
                    keep = False
                    break
            if keep:
                filtered_circles.append(circle)
                center = (circle[0], circle[1])
                radius = circle[2]
                cv2.circle(mask, center, radius, (255), thickness=cv2.FILLED)

    # Application des cercles du masque sur l'image originale
    pieces = cv2.bitwise_and(image_redimensionnee, image_redimensionnee, mask=mask)
    #cv2.imshow("Masque appliqué", resize_for_display(pieces))

    # Afficher le nombre de pièces détectées
    nbPieces = len(filtered_circles) if circles is not None else 0
    #print(f"Nombre de pièces détectées par l'algorithme : {nbPieces}")

    # Dessiner les cercles détectés
    if filtered_circles is not None:
        for circle in filtered_circles:
            center = (circle[0], circle[1])
            radius = circle[2]
            cv2.circle(image_clr_redimensionnee, center, radius, (255), thickness=2)


    # Affichage des cercles détectés
    #cv2.imshow("Cercles détectés par l'algorithme", resize_for_display(image_redimensionnee_copy))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    nbVeriteTerrain = len(image_annotations)

    if nbPieces == nbVeriteTerrain : 
        cv2.imshow("Cercles détectés par l'algorithme", resize_for_display(image_clr_redimensionnee))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return True
    else :
        return False



folder_path = 'C:\\Users\\mmuzz\\Desktop\\Projet Image\\Base\\valid'
# Charger le fichier JSON
with open(folder_path+'\\_annotations.coco.json', 'r') as f:
    data = json.load(f)


# Vérifier si le fichier JSON contient une clé "images"
if 'images' in data:
    images = data['images']
    annotations = data['annotations']
    resultat =[]
    # Boucle sur les éléments "image"
    for image_id in range(len(images)):
        image = images[image_id]
        image_annotations = [annotation for annotation in annotations if annotation.get('image_id') == image_id]
        # Vérifier si l'image contient un "file name"
        if 'file_name' in image:
            file_name = image['file_name']
            image_path = os.path.join(folder_path, file_name)
            resultat.append(process_image(image_path,image_annotations))

else:
    print("Le fichier JSON ne contient pas de clé 'images'.")


# Compter le nombre de True dans la liste
true_count = resultat.count(True)

# Calculer le pourcentage de True
percentage_true = (true_count / len(resultat)) * 100

# Afficher le résultat
print(f"Le pourcentage de valeurs True est de {percentage_true:.2f}%")