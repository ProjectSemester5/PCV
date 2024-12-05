import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, LabelFrame
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os

# Global variables
img_path = None
img_gamma = None
status_kematangan = ""
maturity_persentase = 0.0
img_with_box = None
yolo_label = ""

def adjust_lightness(img_lab, gamma=1.0):
    l_channel, a_channel, b_channel = cv2.split(img_lab)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    l_channel_adjusted = cv2.LUT(l_channel, table)
    img_lab_adjusted = cv2.merge((l_channel_adjusted, a_channel, b_channel))
    return img_lab_adjusted

def draw_bounding_box(img, contours):
    if not contours:
        return img, None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    img_with_box = img.copy()
    cv2.rectangle(img_with_box, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img_with_box, (x, y, w, h)

def calculate_yolo_format(img_shape, bbox):
    if not bbox:
        return None
    img_h, img_w = img_shape[:2]
    x, y, w, h = bbox
    
    # Calculate YOLO format values
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    width = w / img_w
    height = h / img_h
    
    return f"1 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def process_image():
    global img_path, img_gamma, status_kematangan, maturity_persentase, img_with_box, yolo_label
    
    if not img_path:
        messagebox.showerror("Error", "Harap masukkan citra terlebih dahulu.")
        return
    
    img = cv2.imread(img_path)
    img = cv2.resize(img, (300, 300))
    
    # Convert to LAB color space
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Adjust lightness using gamma correction
    img_lab_adjusted = adjust_lightness(img_lab, gamma=1.0)
    img_gamma = cv2.cvtColor(img_lab_adjusted, cv2.COLOR_LAB2BGR)
    
    # Apply red and yellow masking
    mask_red = cv2.inRange(img_lab[:, :, 1], 140, 210)  # 'a' channel (red)
    mask_yellow = cv2.inRange(img_lab[:, :, 2], 165, 200)  # 'b' channel (yellow)
    combined_mask = cv2.bitwise_or(mask_red, mask_yellow)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # Detect edges
    gray = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours and calculate areas
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = sum(cv2.contourArea(cnt) for cnt in contours)
    red_area = cv2.countNonZero(mask_red)
    
    # Calculate maturity percentage
    maturity_persentase = (red_area / total_area) * 100 if total_area > 0 else 0.0
    
    # Classify maturity
    if maturity_persentase >= 80:
        status_kematangan = "Matang"
    elif 20 < maturity_persentase < 80:
        status_kematangan = "Setengah Matang"
    else:
        status_kematangan = "Mentah"
    
    # Draw bounding box and calculate YOLO format label
    img_with_box, bbox = draw_bounding_box(img, contours)
    yolo_label = calculate_yolo_format(img.shape, bbox)
    
    # Display results
    display_results(img, img_gamma, mask_red, edges, cleaned_mask, contours, img_with_box)

def display_results(img, img_gamma, mask_red, edges, cleaned_mask, contours, img_with_box):
    # Clear existing widgets in frame_results
    for widget in frame_results.winfo_children():
        widget.destroy()
    
    # Create Matplotlib figure and subplots
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    ax[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0, 0].set_title("Gambar Asli")
    ax[0, 0].axis("off")
    
    ax[0, 1].imshow(cv2.cvtColor(img_gamma, cv2.COLOR_BGR2RGB))
    ax[0, 1].set_title("Gamma Correction")
    ax[0, 1].axis("off")
    
    ax[0, 2].imshow(mask_red, cmap="gray")
    ax[0, 2].set_title("Masking Merah")
    ax[0, 2].axis("off")
    
    ax[1, 0].imshow(edges, cmap="gray")
    ax[1, 0].set_title("Deteksi Tepi")
    ax[1, 0].axis("off")
    
    ax[1, 1].imshow(cleaned_mask, cmap="gray")
    ax[1, 1].set_title("Masking Bersih")
    ax[1, 1].axis("off")
    
    img_with_contours = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)
    ax[1, 2].imshow(img_with_contours)
    ax[1, 2].set_title("Kontur")
    ax[1, 2].axis("off")
    
    plt.suptitle(f"Kematangan: {status_kematangan} ({maturity_persentase:.2f}%)", fontsize=16)
    plt.tight_layout()
    
    # Display the figure on tkinter canvas
    canvas = FigureCanvasTkAgg(fig, master=frame_results)
    canvas.get_tk_widget().pack(pady=10, padx=10, expand=True)
    canvas.draw()
    
    # Show bounding box image
    display_bounding_box(img_with_box)

def display_bounding_box(img_with_box):
    # Clear previous bounding box and YOLO label info
    for widget in frame_bounding_box.winfo_children():
        widget.destroy()
    
    img_rgb = cv2.cvtColor(img_with_box, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
    
    lbl_bounding_box = tk.Label(frame_bounding_box, image=img_tk)
    lbl_bounding_box.image = img_tk
    lbl_bounding_box.pack()
    
    # Show YOLO label info
    for widget in frame_bounding_box_info.winfo_children():
        widget.destroy()
    
    # Adjust YOLO label based on maturity
    initial_yolo_value = {"Matang": "1", "Mentah": "0", "Setengah Matang": "2"}.get(status_kematangan, "Unknown")
    yolo_label_with_status = f"{initial_yolo_value} {yolo_label[2:]}"
    
    lbl_yolo = tk.Label(frame_bounding_box_info, text=f"YOLO Format Label:\n{yolo_label_with_status}", font=("Arial", 12), padx=10)
    lbl_yolo.pack()

def open_image():
    global img_path, img_tk
    img_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if img_path:
        img = Image.open(img_path).resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        lbl_img.config(image=img_tk)
        lbl_img.image = img_tk
    else:
        messagebox.showerror("Error", "Citra tidak ditemukan.")

# GUI Setup
root = tk.Tk()
root.title("Analisis Kematangan Strawberry")
root.geometry("900x600")

# Canvas for scrollable frame
canvas = tk.Canvas(root)
scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

# Image panel
frame_image = LabelFrame(scrollable_frame, text="Citra Buah", padx=10, pady=10)
frame_image.pack(padx=20, pady=10, fill="x")

lbl_img = tk.Label(frame_image)
lbl_img.pack()

btn_open = tk.Button(frame_image, text="Buka Citra", command=open_image)
btn_open.pack(pady=10)

# Image analysis frame
frame_results = LabelFrame(scrollable_frame, text="Hasil Analisis", padx=10, pady=10)
frame_results.pack(padx=20, pady=10, fill="both", expand=True)

# Bounding box frame
frame_bounding_box = LabelFrame(scrollable_frame, text="Bounding Box", padx=10, pady=10)
frame_bounding_box.pack(padx=20, pady=10, fill="both", expand=True)

# YOLO info frame
frame_bounding_box_info = LabelFrame(scrollable_frame, text="Info YOLO", padx=10, pady=10)
frame_bounding_box_info.pack(padx=20, pady=10, fill="both", expand=True)

# Analyze button
btn_process = tk.Button(frame_image, text="Analisis", command=process_image)
btn_process.pack(pady=10)

# Scrollbar setup
scrollbar.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)

def export_yolo_label():
    global img_path, yolo_label
    
    if not img_path:
        messagebox.showerror("Error", "Harap masukkan citra terlebih dahulu.")
        return
    
    # Menyusun nama file dari path gambar (menggunakan nama file gambar)
    filename = img_path.split("/")[-1].split(".")[0]
    
    # Menentukan folder 'label' dan memastikan folder tersebut ada
    folder_name = "label"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Menyusun nama file output dengan folder 'label'
    output_filename = os.path.join(folder_name, f"{filename}.txt")
    
    # Membuat file .txt dan menulis YOLO label
    with open(output_filename, "w") as file: 
        file.write(yolo_label)
    
    messagebox.showinfo("Export Success", f"File YOLO label berhasil diekspor ke {output_filename}")

# Tambahkan tombol Export di GUI
btn_export = tk.Button(frame_image, text="Export", command=export_yolo_label)
btn_export.pack(pady=10)


root.mainloop()
