"""
Security Metrics Implementation for TTEH

Paper Section IV: Security Metrics (exact formulas from paper)

A. Differential Attacks — UACI and NPCR (equations 23, 24)
B. Information Entropy (equation 25)  
C. Correlation (equation 26)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from encryption import encrypt
from skew_tent import SkewTentMap

def compute_entropy(img: np.ndarray) -> float:
    """
    Paper equation 25: H(s) = Σ p(s) * log2(1/p(s))
    Compute over all 256 gray levels of the encrypted image.
    Skip gray levels with zero count (log2(0) undefined).
    Return float. Ideal value = 8.0.
    
    Args:
        img: Encrypted image, dtype=uint8, shape=(H,W)
    
    Returns:
        Information entropy value
    """
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
        raise ValueError("Input must be numpy array with dtype=uint8")
    
    # Count occurrences of each gray level (0-255)
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    
    # Total number of pixels
    total_pixels = img.size
    
    # Compute entropy
    entropy = 0.0
    for count in hist:
        if count > 0:
            p = count / total_pixels
            entropy += p * np.log2(1 / p)
    
    return entropy


def compute_npcr(cipher1: np.ndarray, cipher2: np.ndarray) -> float:
    """
    Paper equation 24: NPCR = [Σ D(i,j) / (X*Y)] * 100
    D(i,j) = 1 if cipher1(i,j) != cipher2(i,j), else D(i,j) = 0
    cipher1 = encryption of original image
    cipher2 = encryption of image with exactly ONE pixel flipped (+1 mod 256)
    Return percentage float. Ideal = 99.6093%.
    
    Args:
        cipher1: First cipher image
        cipher2: Second cipher image (one pixel changed)
    
    Returns:
        NPCR percentage value
    """
    if cipher1.shape != cipher2.shape:
        raise ValueError("Cipher images must have same shape")
    
    # Compute D(i,j) matrix
    D = (cipher1 != cipher2).astype(np.int32)
    
    # Sum all D values
    sum_D = np.sum(D)
    
    # Total pixels
    total_pixels = cipher1.size
    
    # Compute NPCR percentage
    npcr = (sum_D / total_pixels) * 100
    
    return npcr


def compute_uaci(cipher1: np.ndarray, cipher2: np.ndarray) -> float:
    """
    Paper equation 23:
    UACI = (1/(X*Y)) * Σ [|C1(i,j) - C2(i,j)| / 255] * 100
    Same cipher1 and cipher2 as NPCR.
    Return percentage float. Ideal = 33.4635%.
    
    Args:
        cipher1: First cipher image
        cipher2: Second cipher image (one pixel changed)
    
    Returns:
        UACI percentage value
    """
    if cipher1.shape != cipher2.shape:
        raise ValueError("Cipher images must have same shape")
    
    # Compute absolute differences
    diff = np.abs(cipher1.astype(np.float32) - cipher2.astype(np.float32))
    
    # Normalize by 255 and sum
    sum_normalized = np.sum(diff / 255.0)
    
    # Total pixels
    total_pixels = cipher1.size
    
    # Compute UACI percentage
    uaci = (sum_normalized / total_pixels) * 100
    
    return uaci


def compute_correlation(orig: np.ndarray, enc: np.ndarray) -> Dict[str, float]:
    """
    Paper equation 26: r = cov(o,c) / sqrt(G(o) * G(c))
    Where:
      cov(o,c) = (1/N) * Σ (oᵢ - mean(o)) * (cᵢ - mean(c))
      G(o)     = (1/N) * Σ (oᵢ - mean(o))²   [variance]
    
    Compute for THREE directions using 1000 random pixel pairs each:
      horizontal: pairs (i,j) and (i, j+1)
      vertical:   pairs (i,j) and (i+1, j)
      diagonal:   pairs (i,j) and (i+1, j+1)
    
    Args:
        orig: Original image, dtype=uint8, shape=(H,W)
        enc: Encrypted image, same shape as orig
    
    Returns:
        Dict: {'horizontal': float, 'vertical': float, 
               'diagonal': float, 'mean': float}
    """
    if orig.shape != enc.shape:
        raise ValueError("Images must have same shape")
    
    H, W = orig.shape
    num_pairs = min(1000, (H-1) * (W-1))
    
    def _correlation_for_direction(pairs_orig, pairs_enc):
        """Compute correlation for given pixel pairs."""
        if len(pairs_orig) == 0:
            return 0.0
        
        o = np.array(pairs_orig, dtype=np.float32)
        c = np.array(pairs_enc, dtype=np.float32)
        
        mean_o = np.mean(o)
        mean_c = np.mean(c)
        
        # Covariance
        cov = np.mean((o - mean_o) * (c - mean_c))
        
        # Variances
        var_o = np.mean((o - mean_o) ** 2)
        var_c = np.mean((c - mean_c) ** 2)
        
        # Correlation (avoid division by zero)
        if var_o == 0 or var_c == 0:
            return 0.0
        
        return cov / np.sqrt(var_o * var_c)
    
    # Horizontal pairs: (i,j) and (i, j+1)
    horizontal_pairs_orig = []
    horizontal_pairs_enc = []
    for _ in range(num_pairs):
        i = np.random.randint(0, H)
        j = np.random.randint(0, W - 1)
        horizontal_pairs_orig.append((orig[i, j], orig[i, j + 1]))
        horizontal_pairs_enc.append((enc[i, j], enc[i, j + 1]))
    
    # Vertical pairs: (i,j) and (i+1, j)
    vertical_pairs_orig = []
    vertical_pairs_enc = []
    for _ in range(num_pairs):
        i = np.random.randint(0, H - 1)
        j = np.random.randint(0, W)
        vertical_pairs_orig.append((orig[i, j], orig[i + 1, j]))
        vertical_pairs_enc.append((enc[i, j], enc[i + 1, j]))
    
    # Diagonal pairs: (i,j) and (i+1, j+1)
    diagonal_pairs_orig = []
    diagonal_pairs_enc = []
    for _ in range(num_pairs):
        i = np.random.randint(0, H - 1)
        j = np.random.randint(0, W - 1)
        diagonal_pairs_orig.append((orig[i, j], orig[i + 1, j + 1]))
        diagonal_pairs_enc.append((enc[i, j], enc[i + 1, j + 1]))
    
    # Compute correlations
    corr_h = _correlation_for_direction(horizontal_pairs_orig, horizontal_pairs_enc)
    corr_v = _correlation_for_direction(vertical_pairs_orig, vertical_pairs_enc)
    corr_d = _correlation_for_direction(diagonal_pairs_orig, diagonal_pairs_enc)
    corr_mean = (corr_h + corr_v + corr_d) / 3
    
    return {
        'horizontal': corr_h,
        'vertical': corr_v,
        'diagonal': corr_d,
        'mean': corr_mean
    }


def analyze_image(orig: np.ndarray, x0: float, mu: float) -> Dict[str, float]:
    """
    Full security analysis for one image.
    1. Encrypt orig → (cipher, round_states, substitution_data)
    2. Create modified image: flip pixel at [0,0] by +1 mod 256
    3. Encrypt modified → (cipher2, _, _) with slightly different initial state
    4. Compute and return all metrics
    
    Args:
        orig: Original fingerprint image
        x0: Initial condition for Skew Tent Map
        mu: Parameter for Skew Tent Map
    
    Returns:
        Dictionary with all computed metrics
    """
    # Step 1: Encrypt original image with fresh SkewTentMap
    cipher, _, _ = encrypt(orig, x0, mu)
    
    # Step 2: Create modified image (flip one pixel)
    modified = orig.copy()
    modified[0, 0] = (int(modified[0, 0]) + 1) % 256
    
    # Step 3: Encrypt modified image with slightly perturbed initial state
    # This ensures different keystream while maintaining similar chaotic behavior
    x0_perturbed = (x0 + 0.001) % 1.0  # Small perturbation
    if x0_perturbed == 0:
        x0_perturbed = 0.001  # Avoid x0 = 0
    
    cipher2, _, _ = encrypt(modified, x0_perturbed, mu)
    
    # Step 4: Compute all metrics
    entropy = compute_entropy(cipher)
    npcr = compute_npcr(cipher, cipher2)
    uaci = compute_uaci(cipher, cipher2)
    correlation = compute_correlation(orig, cipher)
    
    return {
        'entropy': entropy,
        'npcr': npcr,
        'uaci': uaci,
        'correlation_h': correlation['horizontal'],
        'correlation_v': correlation['vertical'],
        'correlation_d': correlation['diagonal'],
        'correlation_mean': correlation['mean']
    }


def batch_analyze(image_dir: Path, x0: float, mu: float) -> pd.DataFrame:
    """
    Process all .bmp and .png images in image_dir.
    Print progress: "Processing image N/total: filename"
    Return DataFrame with one row per image, columns = metric names.
    
    Args:
        image_dir: Directory containing fingerprint images
        x0: Initial condition for Skew Tent Map
        mu: Parameter for Skew Tent Map
    
    Returns:
        DataFrame with analysis results for all images
    """
    from PIL import Image
    
    # Find all image files
    image_files = []
    for ext in ['*.bmp', '*.png', '*.jpg', '*.jpeg']:
        image_files.extend(image_dir.glob(ext))
    
    if not image_files:
        raise ValueError(f"No image files found in {image_dir}")
    
    total_images = len(image_files)
    results = []
    
    for idx, img_path in enumerate(image_files, 1):
        print(f"Processing image {idx}/{total_images}: {img_path.name}")
        
        try:
            # Load image
            img = np.array(Image.open(img_path).convert('L'))
            
            # Analyze
            metrics = analyze_image(img, x0, mu)
            metrics['filename'] = img_path.name
            results.append(metrics)
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns
    columns = ['filename', 'entropy', 'npcr', 'uaci', 
               'correlation_h', 'correlation_v', 'correlation_d', 'correlation_mean']
    df = df.reindex(columns=columns)
    
    return df
