"""Entraînement et validation du CNN."""

import argparse
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from vision import CLASSES, DOSSIER_PROJET, MODELE_ROBOT, TAILLE_IMAGE, preparer_image

DONNEES = DOSSIER_PROJET / "image_rgb"
POIDS = MODELE_ROBOT.with_suffix(".pt")
TAUX_APPRENTISSAGE = 0.001
TAILLE_LOT = 32
NOMBRE_EPOQUES = 10


class IA:
    def __init__(self, appareil=None):
        self.appareil = torch.device(appareil or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.modele = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, len(CLASSES)),
        ).to(self.appareil)

    @staticmethod
    def _lots(dossier, melanger):
        banque = ImageFolder(dossier, transform=preparer_image)
        if tuple(banque.classes) != CLASSES:
            raise ValueError(f"{dossier} doit contenir les deux classes {', '.join(CLASSES)}.")
        return DataLoader(banque, batch_size=TAILLE_LOT, shuffle=melanger)

    def entrainer(self, dossier):
        lots = self._lots(dossier, melanger=True)
        critere = nn.CrossEntropyLoss()
        optimiseur = torch.optim.Adam(self.modele.parameters(), lr=TAUX_APPRENTISSAGE)
        total = len(lots.dataset)
        self.modele.train()
        for epoque in range(1, NOMBRE_EPOQUES + 1):
            erreurs = 0.0
            corrects = 0
            for images, labels in lots:
                images, labels = images.to(self.appareil), labels.to(self.appareil)
                optimiseur.zero_grad()
                scores = self.modele(images)
                perte = critere(scores, labels)
                perte.backward()
                optimiseur.step()
                erreurs += perte.item() * labels.size(0)
                corrects += (scores.argmax(1) == labels).sum().item()
            print(f"Époque {epoque:2}/{NOMBRE_EPOQUES} | perte {erreurs / total:.4f} | "f"entraînement {corrects}/{total}", flush=True)
        self.modele.eval()

    @torch.inference_mode()
    def evaluer(self, dossier):
        lots = self._lots(dossier, melanger=False)
        self.modele.eval()
        matrice = torch.zeros((len(CLASSES), len(CLASSES)), dtype=torch.int64)
        for images, labels in lots:
            predictions = self.modele(images.to(self.appareil)).argmax(1).cpu()
            indices = labels * len(CLASSES) + predictions
            matrice += torch.bincount(indices, minlength=len(CLASSES)**2).reshape(matrice.shape)
        return matrice

    @torch.inference_mode()
    def detecter_obstacle(self, image):
        self.modele.eval()
        entree = torch.from_numpy(preparer_image(image)).unsqueeze(0).to(self.appareil)
        classe = self.modele(entree).argmax(1).item()
        return CLASSES[classe] == "arreter"

    def sauvegarder(self, chemin):
        chemin = Path(chemin)
        chemin.parent.mkdir(parents=True, exist_ok=True)
        poids = {nom: tenseur.cpu() for nom, tenseur in self.modele.state_dict().items()}
        torch.save(poids, chemin)

    def charger(self, chemin):
        self.modele.load_state_dict(torch.load(chemin, map_location="cpu", weights_only=True))
        self.modele.eval()

    def exporter_onnx(self, chemin):
        chemin = Path(chemin)
        chemin.parent.mkdir(parents=True, exist_ok=True)
        entree = torch.zeros(1, 3, TAILLE_IMAGE, TAILLE_IMAGE, device=self.appareil)
        self.modele.eval()
        torch.onnx.export(
            self.modele, (entree,), str(chemin),
            input_names=["image"], output_names=["scores"],
            opset_version=11, dynamo=False,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donnees", type=Path, default=DONNEES)
    parser.add_argument("--evaluer", action="store_true", help="Évalue les poids déjà sauvegardés.")
    args = parser.parse_args()

    torch.manual_seed(7)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    ia = IA()
    print(f"Appareil : {ia.appareil}", flush=True)
    if args.evaluer:
        ia.charger(POIDS)
    else:
        ia.entrainer(args.donnees / "entrainement")
        ia.sauvegarder(POIDS)
        ia.charger(POIDS)
        ia.exporter_onnx(MODELE_ROBOT)

    matrice = ia.evaluer(args.donnees / "validation")
    corrects = matrice.trace().item()
    total = matrice.sum().item()
    print(f"Validation : {corrects}/{total} ({corrects / total:.1%})")
    print("                  Prédit arreter    Prédit continuer")
    for classe, ligne in zip(CLASSES, matrice.tolist()):
        print(f"Réel {classe:9} {ligne[0]:>14} {ligne[1]:>19}")


if __name__ == "__main__":
    main()
