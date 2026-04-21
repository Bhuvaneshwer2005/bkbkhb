#!/usr/bin/env python3
"""
TTEH Fingerprint Image Encryption - GUI Launcher
Simple script to launch the TTEH encryption system GUI.
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from gui import main
    print("Starting TTEH Fingerprint Encryption GUI...")
    main()
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required packages: pip install -r requirements.txt")
except Exception as e:
    print(f"Error starting GUI: {e}")
