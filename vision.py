"""Prétraitement commun et détection d'obstacles sur le Raspberry Pi."""

from pathlib import Path
from threading import Event, Thread
from time import monotonic

import numpy as np
from PIL import Image

CLASSES = ("arreter", "continuer")
TAILLE_IMAGE = 128
DOSSIER_PROJET = Path(__file__).resolve().parent
MODELE_ROBOT = DOSSIER_PROJET / "pt_modele" / "kf2_modele.onnx"
DUREE_PREDICTION = 2.0


def preparer_image(image):
    if isinstance(image, Image.Image):
        rgb = image.convert("RGB")
    else:
        if (
            not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3
            or image.dtype != np.uint8 or not image.size
        ):
            raise ValueError("Une image BGR est attendue.")
        rgb = Image.fromarray(image[:, :, ::-1])
    rgb = rgb.resize((TAILLE_IMAGE, TAILLE_IMAGE), Image.BILINEAR)
    pixels = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1)
    return np.ascontiguousarray(pixels) / np.float32(255)


class DetectionObstacle:

    def __init__(self, camera, chemin=MODELE_ROBOT):
        import cv2

        chemin = Path(chemin)
        if not chemin.is_file():
            raise FileNotFoundError(f"Copiez le modèle ONNX sur le Pi : {chemin}")
        self._reseau = cv2.dnn.readNetFromONNX(str(chemin))
        self._camera = camera
        self._arret = Event()
        self._prediction = (None, 0.0)
        self._fil = Thread(target=self._surveiller, name="detection-obstacle", daemon=True)

    def demarrer(self):
        self._fil.start()

    def etat(self):
        obstacle, prise = self._prediction
        if monotonic() - prise > DUREE_PREDICTION:
            return None
        return obstacle

    def _surveiller(self):
        try:
            self._camera.demarrer()
            if self._arret.wait(1.0):
                return
            while not self._arret.is_set():
                prise = monotonic()
                image = self._camera.capturer()
                self._reseau.setInput(preparer_image(image)[None, ...])
                logits = self._reseau.forward()
                if logits.shape != (1, len(CLASSES)) or not np.isfinite(logits).all():
                    raise ValueError("Sortie du modèle d'obstacles invalide.")
                obstacle = CLASSES[int(logits.argmax())] == "arreter"
                self._prediction = obstacle, prise
        except Exception as erreur:
            print(f"Vision interrompue : {erreur}")
        finally:
            self._prediction = (None, 0.0)
            self._camera.arreter()

    def arreter(self):
        self._arret.set()
        if self._fil.ident is not None:
            self._fil.join(timeout=DUREE_PREDICTION)
