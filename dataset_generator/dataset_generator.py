import os
import random as rd
import json
from PIL import Image, ImageEnhance
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Paramètres de transformation
nb_images_par_classe = 1000
nb_image_sans_objet = 200
rotation_max = 180  # degrés
taille_min = 0.03   # pourcentage de la taille originale
taille_max = 0.4    # pourcentage de la taille originale

# Charger les informations liées au objets
with open("objects.json", "r", encoding="utf-8") as f:
    L_objets = json.load(f)


L_background = [
    p for p in Path("dataset_generator/images/backgrounds").rglob("*")
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
]

rd.shuffle(L_background)
L_background_no_objects = L_background[:nb_image_sans_objet]
L_background = L_background[nb_image_sans_objet:]
    
def process_image(objet, j, background, rotation_max, taille_min, taille_max, candidates_cache):
    # Charger et préparer le fond
    fond_path = background
    random_fond = Image.open(fond_path).convert("RGBA")
    random_fond = random_fond.resize((640, 640), Image.Resampling.LANCZOS)

    # Charger et transformer l'objet
    # Utiliser les candidates pré-calculées du cache
    image_dir = objet['path_photo']
    candidates = candidates_cache[image_dir]
    if not candidates:
        raise FileNotFoundError(f"Aucune image trouvée dans le dossier: {image_dir}")
    image_file = rd.choice(candidates)
    image_path = os.path.join(image_dir, image_file)
    image = Image.open(image_path).convert("RGBA")

    # Rotation + redimensionnement
    image = image.rotate(rd.uniform(-rotation_max, rotation_max), expand=True)
    scale_factor = rd.uniform(taille_min, taille_max)
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Placement aléatoire
    dim_fond = random_fond.size
    dim_image = image.size
    x = rd.randint(0, dim_fond[0] - dim_image[0])
    y = rd.randint(0, dim_fond[1] - dim_image[1])

    # Coller l’image
    random_fond.alpha_composite(image, dest=(x, y))

    # Filtre couleur sur toute l'image finale
    arr = np.array(random_fond, dtype=np.float32)  # RGBA → tableau 4D
    r_noise, g_noise, b_noise = [rd.uniform(0.6, 1.4) for _ in range(3)]
    arr[..., 0] *= r_noise  # canal Rouge
    arr[..., 1] *= g_noise  # canal Vert
    arr[..., 2] *= b_noise  # canal Bleu
    # Clamp pour éviter de dépasser 255
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    random_fond = Image.fromarray(arr, mode="RGBA")

    # Sauvegarde
    filename = f"image_{objet['id']}_{j+1}.png"
    output_path = f"dataset_generator/images/dataset/images/{filename}"
    random_fond.save(output_path)

    return filename, x, y, dim_image

def generate_dataset(L_objets, L_background, L_background_no_objects, nb_images_par_classe,
                     rotation_max, taille_min, taille_max):
    train_annotation = {"images": [], "annotations": [], "categories": []}
    test_annotation = {"images": [], "annotations": [], "categories": []}
    train_annotation["categories"] = [{"id": objet['id'], "name": objet['name']} for objet in L_objets]
    test_annotation["categories"] = [{"id": objet['id'], "name": objet['name']} for objet in L_objets]

    os.makedirs("dataset_generator/images/dataset/images", exist_ok=True)
    
    #gestion des images sans objet
    for i, background in enumerate(tqdm(L_background_no_objects, desc="Génération des images sans objet")):
        fond_path = background
        random_fond = Image.open(fond_path).convert("RGBA")
        random_fond = random_fond.resize((640, 640), Image.Resampling.LANCZOS)

        filename = f"image_0_{i+1}.png"
        output_path = f"dataset_generator/images/dataset/images/{filename}"
        random_fond.save(output_path)

        if i < int(nb_image_sans_objet * 0.7):
            img_id = len(train_annotation["images"]) + 1
            train_annotation["images"].append({
                "file_name": filename, "id": img_id, "width": 640, "height": 640
            })
        else:
            img_id = len(test_annotation["images"]) + 1
            test_annotation["images"].append({
                "file_name": filename, "id": img_id, "width": 640, "height": 640
            })


    # Pré-calculer les listes de fichiers candidats pour chaque objet
    candidates_cache = {}
    for objet in L_objets:
        image_dir = objet['path_photo']
        candidates = [f for f in os.listdir(image_dir)
                      if os.path.isfile(os.path.join(image_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not candidates:
            raise FileNotFoundError(f"Aucune image trouvée dans le dossier: {image_dir}")
        candidates_cache[image_dir] = candidates

    # Multi-threading : accélère le traitement des images (I/O bound)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for objet in L_objets:
            for j in range(nb_images_par_classe):
                background = L_background.pop(0)
                L_background.append(background)
                
                futures.append(
                    executor.submit(process_image, objet, j, background,
                                    rotation_max, taille_min, taille_max, candidates_cache)
                )

        for i, f in enumerate(tqdm(futures, desc="Génération des images")):
            filename, x, y, (w, h) = f.result()
            # Déterminer si train ou test
            class_idx = i // nb_images_par_classe
            j = i % nb_images_par_classe
            if j < int(nb_images_par_classe * 0.7):
                img_list, ann_list = train_annotation["images"], train_annotation["annotations"]
            else:
                img_list, ann_list = test_annotation["images"], test_annotation["annotations"]

            img_id = len(img_list) + 1
            ann_id = len(ann_list) + 1
            img_list.append({
                "file_name": filename, "id": img_id, "width": 640, "height": 640
            })
            ann_list.append({
                "image_id": img_id, "category_id": class_idx + 1,
                "bbox": [x, y, w, h], "id": ann_id, "area": w * h, "iscrowd": 0
            })

    # Écriture des fichiers d’annotations
    with open("dataset_generator/images/dataset/train_annotations.json", "w") as f:
        json.dump(train_annotation, f, indent=2)
    with open("dataset_generator/images/dataset/test_annotations.json", "w") as f:
        json.dump(test_annotation, f, indent=2)

    print("Dataset généré avec succès.")

generate_dataset(L_objets, L_background, L_background_no_objects, nb_images_par_classe,
                 rotation_max, taille_min, taille_max)