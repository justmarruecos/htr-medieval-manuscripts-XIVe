"""Pipeline de pretraitement d'images de manuscrits medievaux.

Implemente les trois etapes requises par l'Etape 2 du brief : correction
d'inclinaison (deskew), amelioration du contraste (CLAHE) et binarisation
adaptative (Sauvola). Chaque fonction est independamment testable (formes,
types, plages de valeurs en sortie).
"""

import cv2
import numpy as np
from skimage.filters import threshold_sauvola


def deskew(image: np.ndarray) -> np.ndarray:
    """Corrige l'inclinaison d'une image de manuscrit en niveaux de gris.

    Estime l'angle d'inclinaison via le rectangle englobant minimal des
    pixels sombres (texte), puis applique une rotation inverse.

    Args:
        image: Image en niveaux de gris, tableau 2D (H, W) de type uint8.

    Returns:
        Image redressee, meme forme et meme type que l'entree.

    Raises:
        ValueError: Si l'image n'est pas un tableau 2D.

    Example:
        >>> img = cv2.imread("page.jpg", cv2.IMREAD_GRAYSCALE)
        >>> img_redressee = deskew(img)
    """
    if image.ndim != 2:
        raise ValueError(f"deskew attend une image 2D (H, W), recu shape={image.shape}")

    coords = np.column_stack(np.where(image < 128))
    if coords.size == 0:
        return image.copy()

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle

    h, w = image.shape
    center = (w // 2, h // 2)
    matrice_rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, matrice_rotation, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Ameliore le contraste local d'une image via CLAHE.

    Args:
        image: Image en niveaux de gris, tableau 2D uint8.
        clip_limit: Limite de contraste pour CLAHE (defaut: 2.0).
        tile_size: Taille de la grille de tuiles (defaut: (8, 8)).

    Returns:
        Image avec contraste local ameliore, meme forme/type que l'entree.

    Raises:
        ValueError: Si l'image n'est pas un tableau 2D.

    Example:
        >>> img_clahe = apply_clahe(img, clip_limit=2.0, tile_size=(8, 8))
    """
    if image.ndim != 2:
        raise ValueError(f"apply_clahe attend une image 2D (H, W), recu shape={image.shape}")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(image)


def binarize_sauvola(image: np.ndarray, window_size: int = 25, k: float = 0.2) -> np.ndarray:
    """Binarise une image via un seuillage adaptatif de Sauvola.

    Args:
        image: Image en niveaux de gris, tableau 2D uint8 (idealement
            apres application de apply_clahe).
        window_size: Taille de la fenetre glissante (defaut: 25, doit
            etre impair).
        k: Parametre de sensibilite de Sauvola (defaut: 0.2). Des valeurs
            plus basses (0.1) recuperent davantage les encres palees, au
            prix de plus de bruit sur les zones vides (cf. README :
            k=0.1 pour CREMMA, k=0.2 pour les registres tabulaires).

    Returns:
        Image binaire uint8 (0 ou 255), texte en noir sur fond blanc.

    Raises:
        ValueError: Si l'image n'est pas 2D ou si window_size est pair.

    Example:
        >>> img_bin = binarize_sauvola(img_clahe, window_size=25, k=0.1)
    """
    if image.ndim != 2:
        raise ValueError(f"binarize_sauvola attend une image 2D (H, W), recu shape={image.shape}")
    if window_size % 2 == 0:
        raise ValueError(f"window_size doit etre impair, recu {window_size}")

    thresh = threshold_sauvola(image, window_size=window_size, k=k)
    return (image > thresh).astype(np.uint8) * 255


def preprocess_page(image: np.ndarray, sauvola_k: float = 0.2) -> dict[str, np.ndarray]:
    """Applique le pipeline complet de pretraitement a une page.

    Enchaine dans l'ordre : deskew -> CLAHE -> binarisation Sauvola,
    conformement a l'Etape 2 du brief.

    Args:
        image: Image en niveaux de gris, tableau 2D uint8.
        sauvola_k: Parametre k pour binarize_sauvola (defaut: 0.2).

    Returns:
        Dictionnaire avec les images intermediaires et finale :
        {"original", "deskewed", "clahe", "binary"}.

    Example:
        >>> resultats = preprocess_page(img, sauvola_k=0.1)
        >>> cv2.imwrite("page_binaire.jpg", resultats["binary"])
    """
    deskewed = deskew(image)
    clahe_img = apply_clahe(deskewed)
    binary = binarize_sauvola(clahe_img, k=sauvola_k)
    return {
        "original": image,
        "deskewed": deskewed,
        "clahe": clahe_img,
        "binary": binary,
    }
