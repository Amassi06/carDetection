import math
import cv2
import cvzone
from ultralytics import YOLO


# Charger le modèle YOLO
model = YOLO("../yoloWeights/yolov8l.pt")

# Classes COCO
classNames = [
    "personne", "velo", "voiture", "moto", "avion", "bus", "train", "camion", "bateau",
    "feu de circulation", "borne d'incendie", "panneau stop", "parcmetre", "banc", "oiseau", "chat",
    "chien", "cheval", "mouton", "vache", "elephant", "ours", "zebre", "girafe", "sac a dos", "parapluie",
    "sac a main", "cravate", "valise", "frisbee", "skis", "snowboard", "ballon de sport", "cerf-volant",
    "batte de baseball", "gant de baseball", "skateboard", "surf", "raquette de tennis", "bouteille",
    "verre a vin", "tasse", "fourchette", "couteau", "cuillere", "bol", "banane", "pomme", "sandwich",
    "orange", "brocoli", "carotte", "hot-dog", "pizza", "beignet", "gateau", "chaise", "canape",
    "plante en pot", "lit", "table a manger", "toilettes", "televiseur/moniteur", "ordinateur portable",
    "souris", "telecommande", "clavier", "telephone portable", "micro-ondes", "four", "grille-pain",
    "evier", "refrigerateur", "livre", "horloge", "vase", "ciseaux", "ours en peluche", "seche-cheveux",
    "brosse a dents"
]

# Ouvrir la vidéo
cap = cv2.VideoCapture("Videos/voitures.mp4")


while True:
    success, img = cap.read()
    if not success:
        print("Erreur lors de la lecture de l'image.")
        break

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Coordonnées de la boîte englobante
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Confiance et classe de l'objet
            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Détection de voiture
            if conf > 0.4:  # Augmenter le seuil de confiance pour réduire les fausses détections
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(img, f'{currentClass} {conf * 100}%',
                                   (max(0, min(x1, img.shape[1] - 100)),
                                    max(20, min(y1 - 20, img.shape[0] - 10))),
                                   scale=0.6, thickness=1)

                # Extraction de la plaque (zone en bas du véhicule)
                #plate_x1, plate_y1, plate_x2, plate_y2 = x1, int(y2 - (y2 - y1) * 0.35), x2, y2
                #cv2.rectangle(img, (plate_x1, plate_y1), (plate_x2, plate_y2), (0, 0, 0), 1)


    # Affichage de l'image avec détections
    cv2.imshow("Image", img)
    cv2.waitKey(1)  # Attendre une courte période pour ne pas bloquer la vidéo