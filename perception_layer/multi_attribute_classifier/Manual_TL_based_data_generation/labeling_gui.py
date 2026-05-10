import json
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from pathlib import Path

FIT_OPTS = ["Slim", "Regular", "Oversized"]
STYLE_OPTS = ["Formal", "Casual", "Sport", "Streetwear"]
WEATHER_OPTS = ["Winter (Freezing)", "Transitional (Mild)", "Summer (Hot)"]
FORMALITY_OPTS = ["Black Tie", "Business", "Smart Casual", "Everyday Casual", "Gym/Workout"]

class LabelingGUI:
    def __init__(self, root, data_dir: str):
        self.root = root
        self.root.title("WardrobeGenie Labeler")
        self.data_dir = Path(data_dir)

        # Basic setup
        with open(self.data_dir / "metadata.json", "r") as f:
            self.metadata = json.load(f)

        self.labels_file = self.data_dir / "manual_labels.json"
        self.labels = json.load(open(self.labels_file)) if self.labels_file.exists() else {}
        self.image_files = list(self.metadata.keys())
        self.current_idx = 0

        while self.current_idx < len(self.image_files) and self.image_files[self.current_idx] in self.labels:
            self.current_idx += 1

        self._build_ui()
        self._load_image()

    def _build_ui(self):
        # --- LEFT SIDE: Scrollable Image Area ---
        img_frame = ttk.Frame(self.root)
        img_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        # Create Canvas and Scrollbars
        self.canvas = tk.Canvas(img_frame, width=800, height=800, bg="gray")
        v_scroll = ttk.Scrollbar(img_frame, orient="vertical", command=self.canvas.yview)
        h_scroll = ttk.Scrollbar(img_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        # Layout Canvas and Scrollbars
        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")

        # --- RIGHT SIDE: Controls ---
        self.vars = {
            "fit": tk.StringVar(), "style": tk.StringVar(),
            "weather": tk.StringVar(), "formality": tk.StringVar()
        }

        panel = ttk.Frame(self.root)
        panel.grid(row=0, column=1, sticky="nw", padx=20, pady=20)

        self._create_radio(panel, "Fit", self.vars["fit"], FIT_OPTS, 0)
        self._create_radio(panel, "Style", self.vars["style"], STYLE_OPTS, 1)
        self._create_radio(panel, "Weather", self.vars["weather"], WEATHER_OPTS, 2)
        self._create_radio(panel, "Formality", self.vars["formality"], FORMALITY_OPTS, 3)

        btn_frame = ttk.Frame(panel)
        btn_frame.grid(row=4, column=0, pady=20)
        ttk.Button(btn_frame, text="< Prev", command=self._prev).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Save & Next >", command=self._save_next).pack(side=tk.LEFT, padx=5)

        self.progress_var = tk.StringVar()
        ttk.Label(panel, textvariable=self.progress_var).grid(row=5, column=0)

        # Keybindings
        self.root.bind('<Return>', lambda e: self._save_next())
        self.root.bind('<Left>', lambda e: self._prev())

    def _create_radio(self, parent, title, var, options, row):
        lf = ttk.LabelFrame(parent, text=title)
        lf.grid(row=row, column=0, sticky="ew", pady=5)
        for val in options:
            ttk.Radiobutton(lf, text=val, variable=var, value=val).pack(anchor="w", padx=10, pady=2)

    def _load_image(self):
        if self.current_idx >= len(self.image_files):
            messagebox.showinfo("Done", "All images labeled!")
            return

        img_name = self.image_files[self.current_idx]
        img = Image.open(self.data_dir / img_name)

        # REMOVED: img.thumbnail call so we see full size
        # OPTIONAL: You can still use img.thumbnail if you want to cap the size
        # (e.g., to 2000x2000) so it doesn't crash on massive 50MB files.

        photo = ImageTk.PhotoImage(img)

        # Clear canvas and add new image
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=photo)
        self.canvas.image = photo  # Keep reference

        # Update the scrollable region to the size of the image
        self.canvas.config(scrollregion=(0, 0, img.width, img.height))

        # Reset selection
        saved = self.labels.get(img_name, {})
        for k, v in self.vars.items():
            v.set(saved.get(k, ""))

        self.progress_var.set(f"Progress: {len(self.labels)} / {len(self.image_files)}")

    def _save_next(self):
        if not all(v.get() for v in self.vars.values()):
            messagebox.showwarning("Incomplete", "Select all attributes.")
            return

        self.labels[self.image_files[self.current_idx]] = {k: v.get() for k, v in self.vars.items()}
        with open(self.labels_file, "w") as f:
            json.dump(self.labels, f, indent=2)

        self.current_idx += 1
        self._load_image()

    def _prev(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self._load_image()

if __name__ == "__main__":
    root = tk.Tk()
    # Optional: Start the window maximized
    root.state('zoomed')
    app = LabelingGUI(root, "clip_based_data_generation/manual_labeling_data")
    root.mainloop()