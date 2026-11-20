#!/usr/bin/env python3
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import random
import sys
import time
import uuid

CLASSES = ("arreter", "continuer")
PARTITIONS = ("entrainement", "validation")
RESOLUTION = (640, 480)
REPERTOIRE = Path(__file__).resolve().parent / "image_rgb"
FENETRE = "C: continuer | A: arreter | U: annuler | X: terminer"


class BanqueImages:

    def __init__(self, racine):
        from PIL import Image

        self.racine = Path(racine).resolve()
        self._historique = []
        self._empreintes = {}
        self.comptes = dict.fromkeys(CLASSES, 0)
        for partition in PARTITIONS:
            for classe in CLASSES:
                (self.racine / partition / classe).mkdir(parents=True, exist_ok=True)
        for classe, chemins in self.fichiers().items():
            for chemin in chemins:
                with Image.open(chemin) as image:
                    self._verifier_image(image)
                    empreinte = hashlib.sha256(image.tobytes()).hexdigest()
                if empreinte in self._empreintes:
                    raise ValueError(f"Photos identiques dans la banque: {chemin}")
                self._empreintes[empreinte] = chemin
                self.comptes[classe] += 1

    @staticmethod
    def _verifier_image(image):
        if image.size != RESOLUTION or image.mode != "RGB":
            raise ValueError(
                f"Photo attendue: RGB 640x480; recu: {image.mode} {image.size}"
            )

    def fichiers(self):
        resultat = {}
        for classe in CLASSES:
            chemins = sorted(
                (p for partition in PARTITIONS
                 for p in (self.racine / partition / classe).glob("*.png")),
                key=lambda p: p.name,
            )
            if len({p.name for p in chemins}) != len(chemins):
                raise ValueError(f"Noms de photos repetes dans la classe {classe}")
            resultat[classe] = chemins
        return resultat

    def sauvegarder(self, image_rgb, classe):
        from PIL import Image

        if classe not in CLASSES:
            raise ValueError(f"Classe inconnue: {classe}")
        image = Image.fromarray(image_rgb)
        self._verifier_image(image)
        empreinte = hashlib.sha256(image.tobytes()).hexdigest()
        if empreinte in self._empreintes:
            raise ValueError("Photo identique deja enregistree. Changez de point de vue.")
        date = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        nom = f"image_{date}_{uuid.uuid4().hex[:8]}.png"
        chemin = self.racine / "entrainement" / classe / nom
        with chemin.open("xb") as fichier:
            try:
                image.save(fichier, format="PNG")
            except BaseException:
                fichier.close()
                chemin.unlink(missing_ok=True)
                raise
        self._empreintes[empreinte] = chemin
        self.comptes[classe] += 1
        self._historique.append((chemin, classe, empreinte))
        return chemin

    def annuler(self):
        if not self._historique:
            return None
        chemin, classe, empreinte = self._historique.pop()
        chemin.unlink()
        del self._empreintes[empreinte]
        self.comptes[classe] -= 1
        return chemin

    def repartir(self, graine=7):
        fichiers = self.fichiers()
        if any(len(fichiers[c]) < 5 for c in CLASSES):
            raise ValueError("Il faut au moins 5 photos de chaque classe pour la repartition 80/20.")
        bilan = {p: {} for p in PARTITIONS}
        deplacements = []
        aleatoire = random.Random(graine)
        for classe in CLASSES:
            chemins = list(fichiers[classe])
            aleatoire.shuffle(chemins)
            nb_validation = round(len(chemins) * 0.20)
            for i, chemin in enumerate(chemins):
                partition = "validation" if i < nb_validation else "entrainement"
                destination = self.racine / partition / classe / chemin.name
                deplacements.append((chemin, destination))
            bilan["validation"][classe] = nb_validation
            bilan["entrainement"][classe] = len(chemins) - nb_validation
        for origine, destination in deplacements:
            if origine != destination:
                origine.rename(destination)
        chemins_actualises = dict(deplacements)
        self._empreintes = {
            h: chemins_actualises[p] for h, p in self._empreintes.items()
        }
        self._historique = [
            (chemins_actualises[p], c, h) for p, c, h in self._historique
        ]
        return bilan


class CameraCollecte:

    def __init__(self):
        self.source = None
        try:
            from camera import Camera

            self.source = Camera(largeur=RESOLUTION[0], hauteur=RESOLUTION[1])
            self.source.demarrer()
            fin = time.monotonic() + 2.0
            while time.monotonic() < fin:
                self.lire_rgb()
        except BaseException:
            self.fermer()
            raise

    def lire_rgb(self):
        import cv2

        bgr = self.source.capturer()
        if bgr.shape != (RESOLUTION[1], RESOLUTION[0], 3):
            raise ValueError(
                f"La camera fournit {bgr.shape}; une image 640x480 couleur est attendue."
            )
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def fermer(self):
        if self.source is not None:
            try:
                self.source.arreter()
            finally:
                self.source = None


def progression(banque, objectif):
    return " | ".join(f"{c}: {banque.comptes[c]}/{objectif}" for c in CLASSES)


def traiter_commande(commande, banque, objectif, obtenir_image):
    if commande in ("x", "q", "\x1b"):
        return False
    if commande == "u":
        chemin = banque.annuler()
        print(f"Photo annulee: {chemin.name}" if chemin else "Rien a annuler dans cette session.")
    elif commande in ("a", "c"):
        classe = "arreter" if commande == "a" else "continuer"
        if banque.comptes[classe] >= objectif:
            print(f"Objectif de {objectif} photos atteint pour {classe}.")
        else:
            try:
                chemin = banque.sauvegarder(obtenir_image(), classe)
                print(f"Sauve: {classe}/{chemin.name}")
            except ValueError as erreur:
                print(erreur)
    print(progression(banque, objectif))
    return True


def collecter(camera, banque, objectif):
    import cv2

    print("C: photo continuer | A: photo arreter | U: annuler | X: terminer")
    print(progression(banque, objectif))

    try:
        cv2.namedWindow(FENETRE, cv2.WINDOW_AUTOSIZE)
        while True:
            rgb = camera.lire_rgb()
            apercu = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.rectangle(apercu, (0, 0), (640, 62), (25, 25, 25), -1)
            for y, texte in (
                (24, progression(banque, objectif)),
                (48, "C: continuer | A: arreter | U: annuler | X: terminer"),
            ):
                cv2.putText(apercu, texte, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(FENETRE, apercu)
            touche = cv2.waitKey(30)
            if cv2.getWindowProperty(FENETRE, cv2.WND_PROP_VISIBLE) < 1:
                return
            if touche != -1:
                if not traiter_commande(chr(touche & 0xFF).lower(), banque, objectif, lambda: rgb):
                    return
    finally:
        cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dossier", type=Path, default=REPERTOIRE)
    parser.add_argument("--objectif", type=int, default=100, help="Photos par classe (defaut: 100)")
    parser.add_argument("--graine", type=int, default=7, help="Graine du tirage 80/20")
    parser.add_argument("--finaliser", action="store_true", help="Repartir les photos sans ouvrir la camera")
    args = parser.parse_args(argv)
    if args.objectif < 5:
        parser.error("--objectif doit etre au moins 5 par classe")
    try:
        banque = BanqueImages(args.dossier)
        if not args.finaliser:
            camera = CameraCollecte()
            try:
                collecter(camera, banque, args.objectif)
            except KeyboardInterrupt:
                print("\nCollecte interrompue; photos conservees.")
            finally:
                camera.fermer()
        if any(banque.comptes[c] < args.objectif for c in CLASSES):
            print(f"Collecte incomplete: {progression(banque, args.objectif)}")
        if any(banque.comptes[c] < 5 for c in CLASSES):
            print(f"Photos conservees dans {banque.racine}. Relancez pour continuer.")
            if args.finaliser:
                return 1
            return 0
        bilan = banque.repartir(args.graine)
        for partition, comptes in bilan.items():
            print(f"{partition}: {comptes}")
        print(f"Dossier de photos a transferer: {banque.racine}")
        return 0
    except (ImportError, OSError, RuntimeError, ValueError) as erreur:
        print(f"Erreur: {erreur}", file=sys.stderr)
        print("Les photos deja sauvegardes sont conservees.", file=sys.stderr)
        print("Installer picamera2 et opencv", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
