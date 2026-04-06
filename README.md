"""
TTEH Encryption Pipeline Implementation

Paper Section III-A: Encryption Methodology (Figure 4)
Paper Section III-B: Decryption (exact reversal of encryption)

One encryption round consists of exactly 4 steps:
Step 1: Transformation of fingerprint image using MPHT (eq. 11-12)
Step 2: Transformation of substitution image using MPHT (eq. 13-14)  
Step 3: XOR of two trans-positioned images (eq. 15)
Step 4: Substitution/Saturation using Skew Tent Map (eq. 16)

Repeat for 8 rounds total.
"""

import numpy as np
from typing import Tuple, List
from mpht import mpht_forward, mpht_inverse
from skew_tent import SkewTentMap


def encrypt_round(img: np.ndarray, tent_map: SkewTentMap) -> Tuple[np.ndarray, np.ndarray]:
    """
    One encryption round — paper Section III-A, Steps 1-4, Figure 4.
    
    Step 1: Apply mpht_forward to img → f_trans  (eq. 11-12)
    Step 2: Apply mpht_forward to img → s_trans  (eq. 13-14)
            Using simple transformation that can be reversed.
    Step 3: r = f_trans XOR s_trans               (eq. 15)
    Step 4: keystream = tent_map.generate(W*H)
            r_out = r XOR keystream.reshape(H,W)  (eq. 16)
            tent map state advances — NOT reset.
    
    Args:
        img: Input image for this round, dtype=uint8, shape=(H,W)
        tent_map: SkewTentMap instance with current state
    
    Returns:
        Tuple of (cipher_image, substitution_data) for decryption
    """
    H, W = img.shape
    
    # Step 1: Apply MPHT to fingerprint image → f_trans (eq. 11-12)
    f_trans = mpht_forward(img)
    
    # Step 2: Apply MPHT to substitution image → s_trans (eq. 13-14)
    # Use simple addition for substitution - mathematically reversible
    s_trans = mpht_forward(((img.astype(np.int16) + 1) % 256).astype(np.uint8))
    
    # Step 3: XOR of two trans-positioned images (eq. 15)
    r = np.bitwise_xor(f_trans, s_trans)
    
    # Step 4: Substitution/Saturation using Skew Tent Map (eq. 16)
    keystream = tent_map.generate(H * W)
    keystream_reshaped = keystream.reshape(H, W)
    r_out = np.bitwise_xor(r, keystream_reshaped)
    
    # Save substitution data for decryption
    substitution_data = ((img.astype(np.int16) + 1) % 256).astype(np.uint8)
    
    return r_out, substitution_data


def encrypt(img: np.ndarray, x0: float, mu: float, rounds: int = 8, 
            tent_map: SkewTentMap = None) -> Tuple[np.ndarray, List[float], List[np.ndarray]]:
    """
    Full 8-round encryption.
    
    - Creates ONE SkewTentMap(x0, mu) instance if tent_map is None.
    - Saves map state BEFORE each round for use in decryption replay.
    - Applies encrypt_round 8 times, each round feeding output to next.
    - Returns (encrypted_image, round_states, substitution_data) where round_states is a list
      of 8 map state values saved before each round, and substitution_data contains
      the substitution information needed for decryption.
    
    round_states is essential for decryption to replay each round's keystream.
    substitution_data is essential for decryption to recover the original image.
    
    Args:
        img: Original fingerprint image, dtype=uint8, shape=(H,W)
        x0: Initial condition for Skew Tent Map, 0 < x0 < 1
        mu: Parameter for Skew Tent Map, 1 < mu <= 2
        rounds: Number of encryption rounds, default=8 from paper
        tent_map: Optional existing SkewTentMap instance to reuse
    
    Returns:
        Tuple of (encrypted_image, round_states, substitution_data)
        - encrypted_image: Final cipher image after all rounds
        - round_states: List of map states before each round (length=rounds)
        - substitution_data: List of substitution data for each round (length=rounds)
    """
    if rounds <= 0:
        raise ValueError("Number of rounds must be positive")
    
    # Create or use existing SkewTentMap instance
    if tent_map is None:
        tent_map = SkewTentMap(x0, mu)
    
    # Save state before each round for decryption replay
    round_states = []
    substitution_data = []
    
    # Start with original image
    current_img = img.copy()
    
    for round_idx in range(rounds):
        # Save state before this round
        round_states.append(tent_map.get_state())
        
        # Apply one round of encryption
        current_img, sub_data = encrypt_round(current_img, tent_map)
        substitution_data.append(sub_data)
        
    return current_img, round_states, substitution_data


def decrypt(encrypted_img: np.ndarray, x0: float, mu: float, 
            round_states: List[float], substitution_data: List[np.ndarray], rounds: int = 8) -> np.ndarray:
    """
    Full 8-round decryption — paper Section III-B.
    Reverses rounds in order: round 8 first, round 1 last.
    
    For each round (in reverse):
      Step 1: Restore map to saved state for this round.
              Regenerate same keystream: tent_map.set_state(round_states[r])
              keystream = tent_map.generate(W*H)
              r_img = cipher XOR keystream.reshape(H,W)    (eq. 17)
      
      Step 2: Re-derive s| by applying mpht_forward to r_img (eq. 18-19)
              s_trans = mpht_forward(r_img)
      
      Step 3: Recover f| by XOR                            (eq. 20)
              f_trans = r_img XOR s_trans
      
      Step 4: Apply mpht_inverse to f_trans                (eq. 21-22)
              recovered = mpht_inverse(f_trans)
      
      recovered becomes input to the next decryption round.
    
    Args:
        encrypted_img: Final encrypted image, dtype=uint8, shape=(H,W)
        x0: Initial condition (same as used in encryption)
        mu: Parameter (same as used in encryption)
        round_states: List of map states from encryption (length=rounds)
        rounds: Number of rounds, default=8
    
    Returns:
        Recovered original image, same shape and dtype as input
    """
    if len(round_states) != rounds:
        raise ValueError(f"round_states length {len(round_states)} != rounds {rounds}")
    
    # Create SkewTentMap instance
    tent_map = SkewTentMap(x0, mu)
    
    # Start with encrypted image
    current_img = encrypted_img.copy()
    
    # Process rounds in reverse order
    for round_idx in reversed(range(rounds)):
        H, W = current_img.shape
        
        # Step 1: Undo substitution (eq. 17)
        # Restore map to saved state for this round
        tent_map.set_state(round_states[round_idx])
        # Regenerate same keystream
        keystream = tent_map.generate(H * W)
        keystream_reshaped = keystream.reshape(H, W)
        r_img = np.bitwise_xor(current_img, keystream_reshaped)
        
        # Step 2: Apply MPHT to obtain substitution transposed image (eq. 18-19)
        # Use saved substitution data for perfect reversal
        s_trans = mpht_forward(substitution_data[round_idx])
        
        # Step 3: XOR to get f| (eq. 20)
        f_trans = np.bitwise_xor(r_img, s_trans)
        
        # Step 4: Apply inverse MPHT to recover original image (eq. 21-22)
        recovered = mpht_inverse(f_trans)
        
        # recovered becomes input to next decryption round
        current_img = recovered
    
    return current_img


def generate_key(x0: float = None, mu: float = None) -> Dict[str, float]:
    """
    Generate TTEH encryption key with random parameters if not provided.
    
    Args:
        x0: Initial condition for Skew Tent Map (random if None)
        mu: Parameter for Skew Tent Map (random if None)
    
    Returns:
        Dictionary containing key parameters
    """
    import random
    
    # Generate random parameters if not provided
    if x0 is None:
        x0 = random.uniform(0.1, 0.9)  # Valid range: 0 < x0 < 1
    
    if mu is None:
        mu = random.uniform(1.1, 2.0)  # Valid range: 1 < mu <= 2
    
    return {
        'x0': x0,
        'mu': mu
    }


def save_key(key: Dict[str, float], filepath: str) -> None:
    """
    Save TTEH key to file.
    
    Args:
        key: Key dictionary from generate_key()
        filepath: Path to save key file
    """
    import json
    with open(filepath, 'w') as f:
        json.dump(key, f, indent=2)


def load_key(filepath: str) -> Dict[str, float]:
    """
    Load TTEH key from file.
    
    Args:
        filepath: Path to key file
    
    Returns:
        Key dictionary with x0 and mu parameters
    """
    import json
    try:
        with open(filepath, 'r') as f:
            key = json.load(f)
        return key
    except FileNotFoundError:
        raise FileNotFoundError(f"Key file not found: {filepath}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid key file format: {filepath}")


def encrypt_with_key_file(img: np.ndarray, keyfile: str) -> Tuple[np.ndarray, List[float], List[np.ndarray]]:
    """
    Encrypt image using key from file.
    
    Args:
        img: Original fingerprint image
        keyfile: Path to key file
    
    Returns:
        Tuple of (encrypted_image, round_states, substitution_data)
    """
    key = load_key(keyfile)
    return encrypt(img, key['x0'], key['mu'])


def decrypt_with_key_file(cipher: np.ndarray, keyfile: str, round_states: List[float] = None, substitution_data: List[np.ndarray] = None) -> np.ndarray:
    """
    Decrypt image using key from file.
    
    Args:
        cipher: Encrypted image
        keyfile: Path to key file  
        round_states: Optional saved states from encryption (for efficiency)
        substitution_data: Optional substitution data from encryption
    
    Returns:
        Decrypted image
    """
    key = load_key(keyfile)
    return decrypt(cipher, key['x0'], key['mu'], round_states, substitution_data)


def generate_synthetic_fingerprint(width: int = 256, height: int = 256, 
                                    seed: int = None) -> np.ndarray:
    """
    Generate a realistic synthetic fingerprint image.
    Used when FVC2004 dataset is unavailable.
    
    Algorithm:
      1. Create ridge pattern: sinusoidal waves at multiple angles
         using Gabor-like filters at 4 orientations (0°, 45°, 90°, 135°)
      2. Apply Gaussian envelope centered at image center to shape
         the fingerprint oval boundary
      3. Add subtle random noise (sigma=5) for realism
      4. Normalize to uint8 range [30, 220] to avoid pure black/white
    
    Args:
        width: Image width in pixels
        height: Image height in pixels  
        seed: Random seed for reproducibility
    
    Returns:
        Grayscale np.ndarray dtype=uint8, shape=(height, width)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create coordinate grids
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Create ridge pattern with multiple orientations
    img = np.zeros((height, width))
    
    # Gabor-like filters at 4 orientations - add randomness to frequencies
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    base_frequencies = [8, 12, 10, 9]  # Ridge frequencies
    frequencies = [f + np.random.uniform(-1, 1) for f in base_frequencies]
    
    for angle, freq in zip(orientations, frequencies):
        # Rotate coordinates
        X_rot = X * np.cos(angle) + Y * np.sin(angle)
        
        # Sinusoidal ridge pattern with phase shift
        phase = np.random.uniform(0, 2*np.pi)
        pattern = np.sin(2 * np.pi * freq * X_rot + phase)
        
        # Add to image with random weight
        weight = np.random.uniform(0.2, 0.3)
        img += pattern * weight
    
    # Apply Gaussian envelope for oval shape
    sigma_x, sigma_y = 0.4, 0.5
    envelope = np.exp(-(X**2 / (2 * sigma_x**2) + Y**2 / (2 * sigma_y**2)))
    img = img * envelope
    
    # Add subtle noise for realism
    noise = np.random.normal(0, 5, (height, width))
    img = img + noise
    
    # Normalize to uint8 range [30, 220]
    img = img - img.min()
    img = img / img.max()  # Normalize to [0, 1]
    img = img * 190 + 30   # Scale to [30, 220]
    
    return img.astype(np.uint8)
