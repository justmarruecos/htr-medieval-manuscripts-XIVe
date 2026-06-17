"""Tests unitaires pour src/preprocessing.py (brief, contrainte #5)."""
import numpy as np
import pytest
from src.preprocessing import deskew, apply_clahe, binarize_sauvola, preprocess_page


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def gray_image():
    """Image en niveaux de gris 200x400, texte simulé (pixels sombres)."""
    img = np.ones((200, 400), dtype=np.uint8) * 240
    img[80:120, 50:350] = 30  # ligne de "texte"
    return img

@pytest.fixture
def blank_image():
    """Image entièrement blanche (cas limite : aucun pixel sombre)."""
    return np.ones((100, 200), dtype=np.uint8) * 255


# ── Tests deskew ──────────────────────────────────────────────────────────────

def test_deskew_output_shape(gray_image):
    result = deskew(gray_image)
    assert result.shape == gray_image.shape

def test_deskew_output_dtype(gray_image):
    result = deskew(gray_image)
    assert result.dtype == np.uint8

def test_deskew_blank_image(blank_image):
    """Image sans pixels sombres : doit retourner une copie sans erreur."""
    result = deskew(blank_image)
    assert result.shape == blank_image.shape

def test_deskew_rejects_3d():
    with pytest.raises(ValueError):
        deskew(np.zeros((100, 200, 3), dtype=np.uint8))


# ── Tests apply_clahe ─────────────────────────────────────────────────────────

def test_clahe_output_shape(gray_image):
    result = apply_clahe(gray_image)
    assert result.shape == gray_image.shape

def test_clahe_output_dtype(gray_image):
    result = apply_clahe(gray_image)
    assert result.dtype == np.uint8

def test_clahe_values_in_range(gray_image):
    result = apply_clahe(gray_image)
    assert result.min() >= 0 and result.max() <= 255

def test_clahe_rejects_3d():
    with pytest.raises(ValueError):
        apply_clahe(np.zeros((100, 200, 3), dtype=np.uint8))


# ── Tests binarize_sauvola ────────────────────────────────────────────────────

def test_sauvola_output_shape(gray_image):
    result = binarize_sauvola(gray_image)
    assert result.shape == gray_image.shape

def test_sauvola_output_binary(gray_image):
    """La sortie ne doit contenir que des 0 et des 255."""
    result = binarize_sauvola(gray_image)
    unique = set(np.unique(result))
    assert unique.issubset({0, 255})

def test_sauvola_rejects_even_window(gray_image):
    with pytest.raises(ValueError):
        binarize_sauvola(gray_image, window_size=24)

def test_sauvola_rejects_3d():
    with pytest.raises(ValueError):
        binarize_sauvola(np.zeros((100, 200, 3), dtype=np.uint8))


# ── Tests preprocess_page ─────────────────────────────────────────────────────

def test_preprocess_returns_dict(gray_image):
    result = preprocess_page(gray_image)
    assert isinstance(result, dict)

def test_preprocess_keys(gray_image):
    result = preprocess_page(gray_image)
    assert set(result.keys()) == {"original", "deskewed", "clahe", "binary"}

def test_preprocess_shapes_consistent(gray_image):
    result = preprocess_page(gray_image)
    for key, img in result.items():
        assert img.shape == gray_image.shape, f"Shape incorrecte pour '{key}'"

def test_preprocess_binary_is_binary(gray_image):
    result = preprocess_page(gray_image)
    unique = set(np.unique(result["binary"]))
    assert unique.issubset({0, 255})