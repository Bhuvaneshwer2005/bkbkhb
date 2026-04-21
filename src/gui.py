"""
TTEH GUI Implementation

tkinter GUI with ONE tab:
1. Encrypt / Decrypt - Individual image processing
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from encryption import encrypt, decrypt, generate_synthetic_fingerprint, generate_key, save_key, load_key, encrypt_with_key_file, decrypt_with_key_file
from metrics import analyze_image


class TTEHGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TTEH: Fingerprint Image Encryption")
        self.root.geometry("1400x900")
        
        # Default key parameters from paper
        self.default_mu = 1.9999
        self.default_x0 = 0.3271
        
        # Current images
        self.original_img = None
        self.encrypted_img = None
        self.decrypted_img = None
        self.round_states = None
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_encrypt_decrypt_tab()
    
    def create_encrypt_decrypt_tab(self):
        """Tab 1: Encrypt / Decrypt"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Encrypt / Decrypt")
        
        # Top control bar
        control_frame = ttk.Frame(tab)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(control_frame, text="Load Image", command=self.load_image).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Load Encrypted", command=self.load_encrypted_image).pack(side='left', padx=5)
        
        ttk.Label(control_frame, text="μ:").pack(side='left', padx=(20, 5))
        self.mu_var = tk.StringVar(value=str(self.default_mu))
        ttk.Entry(control_frame, textvariable=self.mu_var, width=10).pack(side='left', padx=5)
        
        ttk.Label(control_frame, text="x₀:").pack(side='left', padx=(10, 5))
        self.x0_var = tk.StringVar(value=str(self.default_x0))
        ttk.Entry(control_frame, textvariable=self.x0_var, width=10).pack(side='left', padx=5)
        
        ttk.Label(control_frame, text="Rounds:").pack(side='left', padx=(10, 5))
        self.rounds_var = tk.StringVar(value="8")
        ttk.Entry(control_frame, textvariable=self.rounds_var, width=5).pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="Encrypt", command=self.encrypt_image).pack(side='left', padx=20)
        ttk.Button(control_frame, text="Decrypt", command=self.decrypt_image).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Export Encrypted", command=self.export_encrypted).pack(side='left', padx=5)
        
        # Key management buttons
        ttk.Button(control_frame, text="Generate Key", command=self.generate_key).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Save Key", command=self.save_key).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Load Key", command=self.load_key).pack(side='left', padx=5)
        
        # Image panels frame
        images_frame = ttk.Frame(tab)
        images_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Three image panels
        self.create_image_panel(images_frame, "Original", 0)
        self.create_image_panel(images_frame, "Encrypted", 1)
        self.create_image_panel(images_frame, "Decrypted", 2)
        
        # Metrics panel
        metrics_frame = ttk.LabelFrame(tab, text="Security Metrics")
        metrics_frame.pack(fill='x', padx=5, pady=5)
        
        self.metrics_labels = {}
        metrics_row = ttk.Frame(metrics_frame)
        metrics_row.pack(pady=5)
        
        for i, metric in enumerate(["Entropy", "NPCR", "UACI", "Correlation"]):
            frame = ttk.Frame(metrics_row)
            frame.pack(side='left', padx=20)
            ttk.Label(frame, text=f"{metric}:", font=('Arial', 10, 'bold')).pack()
            label = ttk.Label(frame, text="--", font=('Arial', 10))
            label.pack()
            self.metrics_labels[metric] = label
        
        # Charts frame
        charts_frame = ttk.Frame(tab)
        charts_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create matplotlib figures
        self.create_charts(charts_frame)
        
        # Status label
        self.status_label = ttk.Label(tab, text="Ready", relief='sunken')
        self.status_label.pack(fill='x', side='bottom', padx=5, pady=5)
    
    def create_image_panel(self, parent, title, column):
        """Create an image display panel"""
        frame = ttk.LabelFrame(parent, text=title)
        frame.grid(row=0, column=column, padx=5, pady=5, sticky='nsew')
        parent.grid_columnconfigure(column, weight=1)
        
        canvas = tk.Canvas(frame, width=300, height=300, bg='gray90')
        canvas.pack(padx=10, pady=10)
        
        # Store canvas reference
        if title == "Original":
            self.original_canvas = canvas
        elif title == "Encrypted":
            self.encrypted_canvas = canvas
        else:
            self.decrypted_canvas = canvas
    
    def create_charts(self, parent):
        """Create matplotlib charts"""
        # Create figure with two subplots
        self.fig = Figure(figsize=(12, 4), dpi=80)
        
        # Histogram subplot
        self.ax_hist = self.fig.add_subplot(121)
        self.ax_hist.set_title("Pixel Histogram")
        self.ax_hist.set_xlabel("Pixel Value")
        self.ax_hist.set_ylabel("Count")
        
        # Correlation subplot  
        self.ax_corr = self.fig.add_subplot(122)
        self.ax_corr.set_title("Pixel Correlation (Horizontal)")
        self.ax_corr.set_xlabel("Pixel Value")
        self.ax_corr.set_ylabel("Adjacent Pixel Value")
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def load_image(self):
        """Load an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Fingerprint Image",
            filetypes=[("Image files", "*.bmp *.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                img = Image.open(file_path).convert('L')
                img = img.resize((256, 256))  # Resize to standard size
                self.original_img = np.array(img, dtype=np.uint8)
                
                # Display image
                self.display_image(self.original_img, self.original_canvas)
                self.status_label.config(text=f"Loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def load_encrypted_image(self):
        """Load an encrypted image and its key file"""
        # First load the encrypted image
        img_path = filedialog.askopenfilename(
            title="Select Encrypted Image",
            filetypes=[("Image files", "*.png *.bmp"), ("All files", "*.*")]
        )
        
        if not img_path:
            return
        
        # Then load the key file
        key_path = filedialog.askopenfilename(
            title="Select Key File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not key_path:
            return
        
        try:
            # Load encrypted image
            img = Image.open(img_path).convert('L')
            img = img.resize((256, 256))
            self.encrypted_img = np.array(img, dtype=np.uint8)
            
            # Load key file with decryption data
            key_data = load_key(key_path)
            
            # Update GUI fields with key parameters
            self.x0_var.set(str(key_data['x0']))
            self.mu_var.set(str(key_data['mu']))
            
            # Load decryption data if available
            if 'round_states' in key_data and 'substitution_data' in key_data:
                self.round_states = key_data['round_states']
                self.substitution_data = key_data['substitution_data']
                self.status_label.config(text=f"Loaded encrypted image and key with decryption data")
            else:
                self.round_states = None
                self.substitution_data = None
                self.status_label.config(text=f"Loaded encrypted image and key (no decryption data - cannot decrypt)")
            
            # Display encrypted image
            self.display_image(self.encrypted_img, self.encrypted_canvas)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load encrypted image/key: {e}")
    
    def encrypt_image(self):
        """Encrypt the loaded image and save both encrypted image and key"""
        if self.original_img is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        def encrypt_thread():
            try:
                # Get parameters
                mu = float(self.mu_var.get())
                x0 = float(self.x0_var.get())
                rounds = int(self.rounds_var.get())
                
                self.status_label.config(text="Encrypting...")
                
                # Generate key with current parameters
                key = generate_key(x0, mu)

                # Encrypt using the generated key values
                self.encrypted_img, self.round_states, self.substitution_data = encrypt(self.original_img, key['x0'], key['mu'], rounds)
                
                # Save key file
                base_name = filedialog.asksaveasfilename(
                    defaultextension=".json",
                    filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                    title="Save Encrypted Image and Key"
                )
                
                if base_name:
                    # Remove extension and use as base name
                    base_path = base_name.rsplit('.', 1)[0] if '.' in base_name else base_name
                    
                    # Save key file with decryption data
                    key_file = f"{base_path}_tteh_key.json"
                    save_key(key, key_file, self.round_states, self.substitution_data)
                    
                    # Save encrypted image
                    img_file = f"{base_path}_encrypted.png"
                    Image.fromarray(self.encrypted_img).save(img_file)
                    
                    # Display encrypted image
                    self.root.after(0, lambda: self.display_image(self.encrypted_img, self.encrypted_canvas))
                    
                    # Compute metrics
                    metrics = analyze_image(self.original_img, x0, mu)
                    
                    # Update metrics display
                    self.root.after(0, lambda: self.update_metrics_display(metrics))
                    
                    # Update charts
                    self.root.after(0, lambda: self.update_charts())
                    
                    success_msg = f"Encryption complete!\n\nSaved files:\n• {key_file}\n• {img_file}"
                    self.root.after(0, lambda: self.status_label.config(text=success_msg))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Encryption failed: {e}"))
                self.root.after(0, lambda: self.status_label.config(text="Encryption failed"))
        
        # Run in separate thread
        thread = threading.Thread(target=encrypt_thread, daemon=True)
        thread.start()
    
    def decrypt_image(self):
        """Decrypt the encrypted image"""
        if self.encrypted_img is None:
            messagebox.showwarning("Warning", "Please load an encrypted image first")
            return
        
        if self.round_states is None or self.substitution_data is None:
            messagebox.showwarning("Warning", "Please load a key file with decryption data first")
            return
        
        def decrypt_thread():
            try:
                # Get parameters
                mu = float(self.mu_var.get())
                x0 = float(self.x0_var.get())
                rounds = int(self.rounds_var.get())
                
                self.status_label.config(text="Decrypting...")
                
                # Decrypt
                self.decrypted_img = decrypt(self.encrypted_img, x0, mu, self.round_states, self.substitution_data, rounds)
                
                # Display decrypted image
                self.root.after(0, lambda: self.display_image(self.decrypted_img, self.decrypted_canvas))
                
                # Verify correctness if original image is available
                if self.original_img is not None:
                    if np.array_equal(self.decrypted_img, self.original_img):
                        self.root.after(0, lambda: self.status_label.config(text="Decryption successful - images match"))
                    else:
                        self.root.after(0, lambda: self.status_label.config(text="Decryption failed - images don't match"))
                else:
                    self.root.after(0, lambda: self.status_label.config(text="Decryption complete (no original to verify)"))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Decryption failed: {e}"))
                self.root.after(0, lambda: self.status_label.config(text="Decryption failed"))
        
        # Run in separate thread
        thread = threading.Thread(target=decrypt_thread, daemon=True)
        thread.start()
    
    def display_image(self, img, canvas):
        """Display image on canvas"""
        # Convert to PIL Image
        pil_img = Image.fromarray(img)
        
        # Resize to fit canvas
        canvas_size = 280
        pil_img = pil_img.resize((canvas_size, canvas_size))
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_img)
        
        # Display on canvas
        canvas.delete("all")
        canvas.create_image(canvas_size//2, canvas_size//2, image=photo)
        canvas.image = photo  # Keep reference
    
    def update_metrics_display(self, metrics):
        """Update metrics labels with color coding"""
        # Color coding ranges from paper
        ranges = {
            'Entropy': (7.99, 8.0),
            'NPCR': (99.5, 100),
            'UACI': (33.4, 33.5),
            'Correlation': (0, 0.01)
        }
        
        for metric, label in self.metrics_labels.items():
            if metric == 'NPCR':
                value = metrics['npcr']
            elif metric == 'UACI':
                value = metrics['uaci']
            elif metric == 'Correlation':
                value = metrics['correlation_mean']
            else:
                value = metrics[metric.lower()]
            
            # Format value
            if metric in ['NPCR', 'UACI']:
                text = f"{value:.2f}%"
            else:
                text = f"{value:.4f}"
            
            label.config(text=text)
            
            # Color coding
            min_val, max_val = ranges[metric]
            if min_val <= value <= max_val:
                label.config(foreground='green')
            else:
                label.config(foreground='orange')
    
    def update_charts(self):
        """Update matplotlib charts"""
        # Clear previous plots
        self.ax_hist.clear()
        self.ax_corr.clear()
        
        # Histogram
        if self.original_img is not None and self.encrypted_img is not None:
            # Original histogram
            hist_orig, bins = np.histogram(self.original_img.flatten(), bins=50, range=(0, 256))
            self.ax_hist.hist(bins[:-1], bins, weights=hist_orig, alpha=0.7, color='blue', label='Original')
            
            # Encrypted histogram
            hist_enc, _ = np.histogram(self.encrypted_img.flatten(), bins=50, range=(0, 256))
            self.ax_hist.hist(bins[:-1], bins, weights=hist_enc, alpha=0.7, color='red', label='Encrypted')
            
            self.ax_hist.set_title("Pixel Histogram")
            self.ax_hist.set_xlabel("Pixel Value")
            self.ax_hist.set_ylabel("Count")
            self.ax_hist.legend()
            self.ax_hist.grid(True, alpha=0.3)
            
            # Correlation scatter plot
            # Sample 1000 random horizontal pairs
            H, W = self.original_img.shape
            num_samples = min(1000, H * (W-1))
            
            indices_i = np.random.randint(0, H, num_samples)
            indices_j = np.random.randint(0, W-1, num_samples)
            
            # Original pairs
            orig_pairs_x = self.original_img[indices_i, indices_j]
            orig_pairs_y = self.original_img[indices_i, indices_j + 1]
            
            # Encrypted pairs
            enc_pairs_x = self.encrypted_img[indices_i, indices_j]
            enc_pairs_y = self.encrypted_img[indices_i, indices_j + 1]
            
            self.ax_corr.scatter(orig_pairs_x, orig_pairs_y, alpha=0.5, s=1, color='blue', label='Original')
            self.ax_corr.scatter(enc_pairs_x, enc_pairs_y, alpha=0.5, s=1, color='red', label='Encrypted')
            
            self.ax_corr.set_title("Pixel Correlation (Horizontal)")
            self.ax_corr.set_xlabel("Pixel Value")
            self.ax_corr.set_ylabel("Adjacent Pixel Value")
            self.ax_corr.legend()
            self.ax_corr.grid(True, alpha=0.3)
        
        # Force canvas redraw
        self.fig.tight_layout()
        self.canvas.draw()
        self.canvas.flush_events()
    
    def generate_key(self):
        """Generate and display random key, update GUI fields"""
        try:
            key = generate_key()  # Generate random parameters
            
            # Update GUI fields with generated values
            self.x0_var.set(f"{key['x0']:.6f}")
            self.mu_var.set(f"{key['mu']:.6f}")
            
            key_text = f"TTEH Random Key Generated:\n\nx0 = {key['x0']:.6f}\nmu = {key['mu']:.6f}\n\nGUI fields updated with new random values.\nSave this key file for encryption/decryption.\n\nRandom parameters provide better security than fixed values."
            
            messagebox.showinfo("Random Key Generated", key_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Key generation failed: {e}")
    
    def save_key(self):
        """Save current key to file"""
        try:
            mu = float(self.mu_var.get())
            x0 = float(self.x0_var.get())
            
            key = generate_key(x0, mu)
            
            filepath = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filepath:
                save_key(key, filepath)
                messagebox.showinfo("Success", f"Key saved to {filepath}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Key save failed: {e}")
    
    def load_key(self):
        """Load key from file with decryption data"""
        try:
            filepath = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filepath:
                key_data = load_key(filepath)
                self.x0_var.set(str(key_data['x0']))
                self.mu_var.set(str(key_data['mu']))
                
                # Load decryption data if available
                if 'round_states' in key_data and 'substitution_data' in key_data:
                    self.round_states = key_data['round_states']
                    self.substitution_data = key_data['substitution_data']
                    messagebox.showinfo("Success", f"Key loaded from {filepath}\n\nDecryption data included.")
                else:
                    messagebox.showinfo("Success", f"Key loaded from {filepath}\n\nNote: No decryption data - cannot decrypt previously encrypted images.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Key load failed: {e}")
    
    def export_encrypted(self):
        """Export encrypted image to file"""
        if self.encrypted_img is None:
            messagebox.showwarning("Warning", "Please encrypt an image first")
            return
        
        try:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("BMP files", "*.bmp"), ("All files", "*.*")]
            )
            
            if filepath:
                Image.fromarray(self.encrypted_img).save(filepath)
                messagebox.showinfo("Success", f"Encrypted image saved to {filepath}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")


def main():
    root = tk.Tk()
    app = TTEHGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
