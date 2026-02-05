import numpy as np
import pyrender
import trimesh
from PIL import Image
import os
from pathlib import Path

def extract_images_from_glb(glb_file,
                            output_dir="renders",
                            resolution=1024,
                            horizontal_steps=60,
                            vertical_steps=20,
                            radius=0.2,
                            height_angle_range=(5, 90),
                            fit_image=True):

    # Créer le répertoire de sortie s'il n'existe pas
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Charger le modèle 3D
    print(f"Chargement du fichier: {glb_file}")
    loaded_obj = trimesh.load(glb_file)
    
    # Si c'est une scène, fusionner tous les meshes
    if isinstance(loaded_obj, trimesh.Scene):
        geom = loaded_obj.to_geometry()
        if isinstance(geom, dict):
            # Concatenate all geometry parts into a single Trimesh
            trimesh_obj = trimesh.util.concatenate(tuple(geom.values()))
        else:
            trimesh_obj = geom
    
    # Convertir le trimesh en pyrender mesh
    mesh = pyrender.Mesh.from_trimesh(trimesh_obj)
    
    # Calculer le centre et la taille de l'objet
    # Centre de l'objet
    centroid = trimesh_obj.centroid
    
    # Créer une scène pyrender
    scene = pyrender.Scene()
    scene.add(mesh)
    # Demander un fond transparent pour la scène
    scene.bg_color = np.array([0.0, 0.0, 0.0, 0.0])
    
    # Ajouter plusieurs lumières ponctuelles pour un éclairage équilibré
    # Lumière au-dessus
    light_top_pos = np.array([centroid[0], centroid[1] + radius * 3.0, centroid[2]])
    light_top = pyrender.PointLight(color=np.ones(3), intensity=0.5)
    light_top_node = pyrender.Node(light=light_top, translation=light_top_pos)
    scene.add_node(light_top_node)

    # Lumières latérales (gauche et droite) pour éviter l'éclairage d'un seul côté
    left_pos = np.array([centroid[0] + radius * 3.0, centroid[1], centroid[2]])
    right_pos = np.array([centroid[0] - radius * 3.0, centroid[1], centroid[2]])
    left_light = pyrender.PointLight(color=np.ones(3), intensity=1.0)
    right_light = pyrender.PointLight(color=np.ones(3), intensity=1.0)
    scene.add_node(pyrender.Node(light=left_light, translation=left_pos))
    scene.add_node(pyrender.Node(light=right_light, translation=right_pos))
    
    # Générer les images
    print(f"Generating {vertical_steps} vertical steps with {horizontal_steps} images per circle...")
    
    frame_counter = 0
    # Créer un renderer offscreen réutilisable
    renderer = pyrender.OffscreenRenderer(viewport_width=resolution, viewport_height=resolution)
    
    # Parcourir les étapes en hauteur
    for step in range(vertical_steps):
        # Calculer l'angle en hauteur pour cette étape
        height_angle = np.interp(
            step,
            [0, vertical_steps - 1],
            [height_angle_range[1], height_angle_range[0]]
        )
        height_rad = np.radians(height_angle)
        
        # Faire un cercle complet à cette hauteur
        for i in range(horizontal_steps):
            # Calculer l'angle azimutal (autour de l'objet)
            azimuth_angle = (i / horizontal_steps) * 2 * np.pi
            azimuth_rad = azimuth_angle
            
            # Calculer la position de la caméra sur la sphère
            x = centroid[0] + radius * np.cos(azimuth_rad) * np.cos(height_rad)
            y = centroid[1] + radius * np.sin(height_rad)
            z = centroid[2] + radius * np.sin(azimuth_rad) * np.cos(height_rad)
            camera_pos = np.array([x, y, z])
        
            # Créer une caméra pointant vers le centre de l'objet
            camera = pyrender.PerspectiveCamera(
                yfov=np.pi / 3.0,  # Champ de vision de 60 degrés
                aspectRatio=1.0
            )
        
            # Créer une pose de caméra pointant vers le centre de l'objet
            direction = centroid - camera_pos
            direction = direction / np.linalg.norm(direction)
        
            # Créer une matrice de rotation pour que la caméra regarde vers le centre
            up = np.array([0, 1, 0])
            right = np.cross(direction, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, direction)
            up = up / np.linalg.norm(up)
        
            rotation_matrix = np.array([
                right,
                up,
                -direction  # Négatif car pyrender utilise la convention -Z forward
            ]).T
        
            pose = np.eye(4)
            pose[:3, :3] = rotation_matrix
            pose[:3, 3] = camera_pos
        
            # Ajouter la caméra à la scène
            camera_node = scene.add(camera, pose=pose)
        
            # Rendu de l'image avec canal alpha (RGBA)
            color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        
            # Convertir en image PIL et sauvegarder
            image = Image.fromarray(color)

            # Si fit_image, crop to object alpha bbox, scale so largest side == resolution,
            if fit_image:
                alpha = image.split()[-1]
                bbox = alpha.getbbox()
                cropped = image.crop(bbox)
                w, h = cropped.size
                scale = float(resolution) / float(max(w, h))
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                resized = cropped.resize((new_w, new_h), resample=Image.LANCZOS)
                final_image = resized
            else:
                final_image = image

            output_path = os.path.join(output_dir, f"render_{frame_counter:04d}.png")
            final_image.save(output_path)
            print(f"Image {frame_counter+1} (vertical {step+1}/{vertical_steps}, horizontal {i+1}/{horizontal_steps})")
        
            # Supprimer la caméra de la scène pour la prochaine itération
            scene.remove_node(camera_node)
            frame_counter += 1

    # Nettoyer le renderer
    renderer.delete()


def main():
    glb_files = list(Path("dataset_generator/images/objects_3d_scan").glob("*.glb"))
    
    for glb_file in glb_files:
        print(f"Treating: {glb_file}")
        output_dir = Path("dataset_generator/images/objects_images") / glb_file.stem
        if output_dir.exists():
            print(f"Skipping : {glb_file} (output directory already exists)")
            continue
        extract_images_from_glb(str(glb_file), output_dir=output_dir)


if __name__ == "__main__":
    main()
