#!/usr/bin/env python3
import os
import sys
import numpy as np
import tifffile as tiff
import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector


class TiffViewerTk:

  def __init__(self, root, initial_path=None):
    self.root = root
    self.root.title("16-bit TIFF Viewer & Histogram (Tkinter Drag & Drop)")
    self.root.geometry("1100x550")

    # Set up Matplotlib Figure and Tk Canvas
    self.figure = Figure(figsize=(12, 5))
    self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
    self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    self.ax_img = self.figure.add_subplot(121)
    self.ax_hist = self.figure.add_subplot(122)
    self.figure.subplots_adjust(bottom=0.15, wspace=0.25)

    self.img = None
    self.is_rgb = False
    self.rect_selector = None

    # Register window for file drops
    self.root.drop_target_register(DND_FILES)
    self.root.dnd_bind("<<Drop>>", self.on_drop)

    if initial_path and os.path.isfile(initial_path):
      self.load_image(initial_path)
    else:
      self.show_placeholder()

  def show_placeholder(self):
    self.ax_img.clear()
    self.ax_hist.clear()
    self.ax_img.text(
        0.5,
        0.5,
        "Drag & Drop a 16-bit TIFF here\nor pass file via command line",
        ha="center",
        va="center",
        fontsize=13,
        color="gray",
        transform=self.ax_img.transAxes,
    )
    self.ax_img.axis("off")
    self.ax_hist.axis("off")
    self.canvas.draw()

  def on_drop(self, event):
    path = event.data.strip("{}")  # Strip Tk path braces if present
    if os.path.isfile(path):
      self.load_image(path)

  def load_image(self, file_path):
    try:
      self.img = tiff.imread(file_path)
    except Exception as e:
      print(f"Failed to read {file_path}: {e}")
      return

    self.root.title(f"TIFF Viewer - {os.path.basename(file_path)}")
    print(f"Loaded: {file_path}")
    print(
        f"Shape: {self.img.shape} | Dtype: {self.img.dtype} | Range:"
        f" [{self.img.min()}, {self.img.max()}]"
    )

    self.is_rgb = self.img.ndim == 3 and self.img.shape[-1] in (3, 4)
    self.render_image_and_histogram()

  def render_image_and_histogram(self):
    self.ax_img.clear()
    self.ax_hist.axis("on")

    if self.is_rgb:
      display_img = (self.img[..., :3] / 256).astype(np.uint8)
      self.ax_img.imshow(display_img)
    else:
      vmin, vmax = np.percentile(self.img, [0.5, 99.5])
      self.ax_img.imshow(
          self.img, cmap="gray", vmin=vmin, vmax=max(vmax, vmin + 1)
      )

    self.ax_img.set_title("Drag to select region (Click to reset)")
    self.ax_img.axis("off")

    self.rect_selector = RectangleSelector(
        self.ax_img,
        self.on_select,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True,
    )

    self.update_histogram(self.img, title="Full Image Histogram")

  def update_histogram(self, data, title="Full Image Histogram"):
    self.ax_hist.clear()
    self.ax_hist.set_yscale("log")

    max_count = 1
    if self.is_rgb:
      colors = ("red", "green", "blue")
      for i, color in enumerate(colors):
        channel_data = data[..., i].ravel()
        hist, edges = np.histogram(channel_data, bins=256, range=(0, 65536))
        max_count = max(max_count, hist.max())
        self.ax_hist.step(
            edges[:-1],
            np.maximum(hist, 0.5),
            color=color,
            alpha=0.8,
            label=color.upper(),
            where="post",
        )
      self.ax_hist.legend(loc="upper right")
    else:
      hist, edges = np.histogram(data.ravel(), bins=256, range=(0, 65536))
      max_count = max(max_count, hist.max())
      self.ax_hist.step(
          edges[:-1],
          np.maximum(hist, 0.5),
          color="black",
          lw=1,
          where="post",
      )
      self.ax_hist.fill_between(
          edges[:-1],
          np.maximum(hist, 0.5),
          0.5,
          step="post",
          color="gray",
          alpha=0.3,
      )

    self.ax_hist.set_title(title)
    self.ax_hist.set_xlabel("16-bit Intensity (0 - 65535)")
    self.ax_hist.set_ylabel("Pixel Count (Log Scale)")
    self.ax_hist.set_xlim(0, 65535)
    self.ax_hist.set_ylim(bottom=0.8, top=max(max_count * 1.5, 10))
    self.ax_hist.grid(True, which="both", linestyle="--", alpha=0.5)

    self.canvas.draw_idle()

  def on_select(self, eclick, erelease):
    if self.img is None:
      return

    x1, y1 = int(round(eclick.xdata)), int(round(eclick.ydata))
    x2, y2 = int(round(erelease.xdata)), int(round(erelease.ydata))

    xmin, xmax = max(0, min(x1, x2)), min(self.img.shape[1], max(x1, x2))
    ymin, ymax = max(0, min(y1, y2)), min(self.img.shape[0], max(y1, y2))

    if xmax - xmin < 2 or ymax - ymin < 2:
      self.update_histogram(self.img, title="Full Image Histogram")
      return

    roi = self.img[ymin:ymax, xmin:xmax]
    title = (
        f"ROI [{xmin}:{xmax}, {ymin}:{ymax}] (Min: {roi.min()}, Max:"
        f" {roi.max()})"
    )
    self.update_histogram(roi, title=title)


if __name__ == "__main__":
  root = TkinterDnD.Tk()
  initial_file = sys.argv[1] if len(sys.argv) > 1 else None
  app = TiffViewerTk(root, initial_path=initial_file)
  root.mainloop()
