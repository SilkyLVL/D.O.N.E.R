# 🦜 Invasive Parrot Detection & Automated Trap System

An automated computer vision monitoring system designed to detect invasive parrots using custom-trained YOLOv5 models, trigger a physical trap latch, and send real-time HTML incident reports with photographic evidence via email.

## 🚀 Features
- Custom-trained YOLOv5 object detector (`best.pt`).
- Graphical user interface (GUI) built with Tkinter.
- Live webcam monitoring with multi-frame confirmation to prevent false triggers.
- Photo upload testing tool.
- Automated email alerts with structured HTML metadata and captured evidence attached.

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd <repo-folder>