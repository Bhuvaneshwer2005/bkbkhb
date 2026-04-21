"""
Chaotic Skew Tent Map Implementation

Paper equation 16:
  x_{n+1} = f_μ(x_n) = {
      μ * x_n,         if x_n < 1/2
      μ * (1 - x_n),   if x_n >= 1/2
  }

Key parameter constraints from paper:
  1 < μ ≤ 2
  0 < x_0 < 1

Stateful across all 8 rounds - never reset between rounds.
"""

import numpy as np


class SkewTentMap:
    """
    Stateful Chaotic Skew Tent Map — paper equation 16.
    State persists across all 8 rounds (never reset between rounds).
    
    x_{n+1} = mu * x_n        if x_n < 0.5
    x_{n+1} = mu * (1 - x_n)  if x_n >= 0.5
    """
    
    def __init__(self, x0: float, mu: float):
        """
        Initialize Skew Tent Map with parameters.
        
        Args:
            x0: initial condition, must satisfy 0 < x0 < 1
            mu: map parameter, must satisfy 1 < mu <= 2
        
        Raises:
            ValueError: if constraints violated.
        
        Runs 200 warm-up iterations on init to escape transient behavior.
        Note: warm-up count is implementation convention, not stated in paper.
        """
        if not (0 < x0 < 1):
            raise ValueError(f"x0 must satisfy 0 < x0 < 1, got {x0}")
        
        if not (1 < mu <= 2):
            raise ValueError(f"mu must satisfy 1 < mu <= 2, got {mu}")
        
        self.x0 = x0
        self.mu = mu
        self.x = x0
        
        # Warm-up iterations to escape transient behavior
        for _ in range(200):
            self.next_value()
    
    def next_value(self) -> float:
        """
        Advance map by one step, return x_{n+1} as float in (0,1).
        
        Implements paper equation 16:
          x_{n+1} = mu * x_n        if x_n < 0.5
          x_{n+1} = mu * (1 - x_n)  if x_n >= 0.5
        """
        if self.x < 0.5:
            self.x = self.mu * self.x
        else:
            self.x = self.mu * (1 - self.x)
        
        # Ensure result stays in (0,1) due to floating point precision
        self.x = max(1e-10, min(1 - 1e-10, self.x))
        
        return self.x
    
    def generate(self, length: int) -> np.ndarray:
        """
        Generate `length` keystream values.
        
        Args:
            length: Number of keystream values to generate
        
        Returns:
            numpy array of uint8 values (0-255)
        
        Scale to uint8: val = int(x * 256) & 0xFF
        Map state is updated — subsequent calls continue from current state.
        """
        if length <= 0:
            return np.array([], dtype=np.uint8)
        
        values = np.zeros(length, dtype=np.uint8)
        
        for i in range(length):
            self.next_value()
            # Scale to uint8 range [0, 255]
            values[i] = int(self.x * 256) & 0xFF
        
        return values
    
    def get_state(self) -> float:
        """
        Return current map value (for saving state before round).
        
        Returns:
            Current state x_n
        """
        return self.x
    
    def set_state(self, x: float):
        """
        Restore map to a previous state (for decryption replay).
        
        Args:
            x: State value to restore, must satisfy 0 < x < 1
        """
        if not (0 < x < 1):
            raise ValueError(f"State must satisfy 0 < x < 1, got {x}")
        
        self.x = x
