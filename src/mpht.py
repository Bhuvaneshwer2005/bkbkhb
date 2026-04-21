"""
Modified Pseudo Hadamard Transform (MPHT) Implementation

Paper equations:
- Forward: eq. 11-14
- Inverse: eq. 21-22

Constants from paper:
- p = 10 (rho constant)
- n = 8 (bit depth)
- mod = 2^8 = 256
"""

import numpy as np


def mpht_forward(img: np.ndarray, p: int = 10, n: int = 8) -> np.ndarray:
    """
    Apply forward MPHT row-by-row to every adjacent pixel pair.
    Paper equations 11-12 (fingerprint) and 13-14 (substitution):
      alpha = (a + b + p) mod 2^n
      beta  = (a + 2b + p) mod 2^n
    
    Args:
        img: Input grayscale image, dtype=uint8, shape=(H, W)
        p: MPHT constant (rho), default=10 from paper eq. 12, 14, 19
        n: Bit depth, default=8 from paper
    
    Returns:
        Transposed image with same shape and dtype as input
    
    Works on grayscale uint8 images of any size.
    If image width is odd, last column is left unchanged.
    Uses numpy vectorized operations — no pixel-level Python loops.
    """
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
        raise ValueError("Input must be numpy array with dtype=uint8")
    
    if img.ndim != 2:
        raise ValueError("Input must be 2D grayscale image")
    
    H, W = img.shape
    result = img.copy()
    
    if W < 2:
        return result
    
    # Extract even and odd columns
    a = img[:, 0::2].astype(np.int32)   # even columns (0, 2, 4, ...)
    b = img[:, 1::2].astype(np.int32)   # odd columns (1, 3, 5, ...)
    
    # Apply MPHT equations
    mod = 2 ** n
    alpha = (a + b + p) % mod
    beta = (a + 2 * b + p) % mod
    
    # Assign back to result array
    result[:, 0::2] = alpha.astype(np.uint8)
    result[:, 1::2] = beta.astype(np.uint8)
    
    # If width is odd, last column remains unchanged (already copied)
    
    return result


def mpht_inverse(img: np.ndarray, p: int = 10, n: int = 8) -> np.ndarray:
    """
    Apply inverse MPHT row-by-row to every adjacent pixel pair.
    Paper equations 21-22:
      b = (beta - alpha) mod 2^n
      a = (2*alpha - beta - p) mod 2^n
    
    Args:
        img: Input transposed image, dtype=uint8, shape=(H, W)
        p: MPHT constant (rho), default=10 from paper eq. 21-22
        n: Bit depth, default=8 from paper
    
    Returns:
        Recovered original image with same shape and dtype as input
    
    Uses numpy vectorized operations — no pixel-level Python loops.
    """
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
        raise ValueError("Input must be numpy array with dtype=uint8")
    
    if img.ndim != 2:
        raise ValueError("Input must be 2D grayscale image")
    
    H, W = img.shape
    result = img.copy()
    
    if W < 2:
        return result
    
    # Extract even and odd columns (these are alpha and beta)
    alpha = img[:, 0::2].astype(np.int32)   # even columns
    beta = img[:, 1::2].astype(np.int32)    # odd columns
    
    # Apply inverse MPHT equations
    mod = 2 ** n
    b = (beta - alpha) % mod
    a = (2 * alpha - beta - p) % mod
    
    # Assign back to result array
    result[:, 0::2] = a.astype(np.uint8)
    result[:, 1::2] = b.astype(np.uint8)
    
    # If width is odd, last column remains unchanged (already copied)
    
    return result
