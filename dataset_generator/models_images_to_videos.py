import cv2
import os
from pathlib import Path
import numpy as np

def create_video_from_renders(renders_dir, output_video="video/output.mp4", fps=30, frame_width=1024, frame_height=1024):
    """
    Crée une vidéo à partir des images PNG dans le répertoire renders.
    
    Args:
        renders_dir (str): Chemin vers le répertoire contenant les images PNG
        output_video (str): Nom du fichier vidéo de sortie
        fps (int): Nombre d'images par seconde
        frame_width (int): Largeur de la vidéo
        frame_height (int): Hauteur de la vidéo
    """
    
    # Vérifier que le répertoire existe
    if not os.path.isdir(renders_dir):
        print(f"Erreur: Le répertoire {renders_dir} n'existe pas")
        return
    
    # Récupérer toutes les images PNG et les trier
    image_files = sorted([f for f in os.listdir(renders_dir) if f.endswith('.png')])
    
    if not image_files:
        print(f"Erreur: Aucune image PNG trouvée dans {renders_dir}")
        return
    
    print(f"Trouvé {len(image_files)} images")
    
    # Définir le codec et créer le writer vidéo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    
    print(f"Création de la vidéo: {output_video}")
    print(f"FPS: {fps}, Résolution: {frame_width}x{frame_height}")
    
    # Ajouter chaque image à la vidéo
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(renders_dir, image_file)
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"Attention: Impossible de lire {image_file}")
            continue
        
        # Redimensionner si nécessaire
        frame = cv2.resize(frame, (frame_width, frame_height))
        
        # Convertir BGR (OpenCV) en RGB si nécessaire (les images PNG sont généralement en RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        out.write(frame)
        
        if (i + 1) % 10 == 0:
            print(f"Traitée {i + 1}/{len(image_files)} images")
    
    out.release()
    print(f"Vidéo créée avec succès: {output_video}")


def main():
    # Chercher les dossiers renders/*/
    renders_base = Path("dataset_generator/images/models_images")
    
    if not renders_base.exists():
        print("Erreur: Le répertoire 'renders' n'existe pas")
        return
    
    # Trouver tous les sous-dossiers contenant des images
    render_dirs = [d for d in renders_base.iterdir() if d.is_dir()]
    
    if not render_dirs:
        print("Aucun sous-dossier trouvé dans 'renders'")
        return
    
    for render_dir in render_dirs:
        print(f"\nTraitement du dossier: {render_dir}")
        
        # Créer une vidéo pour chaque dossier
        output_video = f"dataset_generator/video/models_videos/{render_dir.stem}_video.mp4"
        create_video_from_renders(str(render_dir), output_video=output_video)


if __name__ == "__main__":
    main()
