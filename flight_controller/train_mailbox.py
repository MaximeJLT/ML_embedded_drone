from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_YAML = PROJECT_ROOT / "datasets" / "mailbox" / "data.yaml"

BASE_MODEL = "yolo11n.pt"

EPOCHS   = 60     
IMG_SIZE = 640   
BATCH    = 16      
DEVICE   = 0      

RUN_PROJECT = PROJECT_ROOT / "runs"
RUN_NAME    = "mailbox_plomberie"

def main():
  
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml doesnt exists : {DATA_YAML}")

    model = YOLO(BASE_MODEL)

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        project=str(RUN_PROJECT),
        name=RUN_NAME,
        exist_ok=True,   
    )

    best = RUN_PROJECT / RUN_NAME / "weights" / "best.pt"
    print("\n" + "=" * 60)
    if best.exists():
        print(f"OK  best.pt genere ici :\n    {best}")
        print("C'est ce fichier qu'on branchera dans NN.py.")
    else:
        print("best.pt introuvable : regarde les logs d'entrainement plus haut.")
    print("=" * 60)


if __name__ == "__main__":
    main()
