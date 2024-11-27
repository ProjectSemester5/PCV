import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, LabelFrame
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

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
    
    # Hitung nilai YOLO format
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
    
    # Konversi ke LAB
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(img_lab)
    
    # Rata-rata intensitas kanal L
    mean_intensity = np.mean(l_channel)
    target_intensity = 168  # intensitas target
    gamma_suggested = max(0.7, min(1, target_intensity))
    
    # Terapkan penyesuaian gamma
    img_lab_adjusted = adjust_lightness(img_lab, gamma=gamma_suggested)
    img_gamma = cv2.cvtColor(img_lab_adjusted, cv2.COLOR_LAB2BGR)
    
    # Masking merah dan kuning
    mask_red = cv2.inRange(a_channel, 140, 210)
    mask_yellow = cv2.inRange(b_channel, 165, 200)
    combined_mask = cv2.bitwise_or(mask_red, mask_yellow)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # Deteksi tepi
    gray = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Kontur dan luas area
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = sum(cv2.contourArea(cnt) for cnt in contours)
    red_area = cv2.countNonZero(mask_red)
    
    if total_area > 0:
        maturity_persentase = (red_area / total_area) * 100
    else:
        maturity_persentase = 0.0
    
    # Klasifikasi kematangan
    if maturity_persentase >= 80:
        status_kematangan = "Matang"
    elif 20 < maturity_persentase < 80:
        status_kematangan = "Setengah Matang"
    else:
        status_kematangan = "Mentah"
    
    # Gambar bounding box dan hitung YOLO format
    img_with_box, bbox = draw_bounding_box(img, contours)
    yolo_label = calculate_yolo_format(img.shape, bbox)
    
    # Tampilkan hasil
    display_results(img, img_gamma, mask_red, edges, cleaned_mask, contours, img_with_box)


def display_results(img, img_gamma, mask_red, edges, cleaned_mask, contours, img_with_box):
    for widget in frame_results.winfo_children():
        widget.destroy()
    
    # Buat figure matplotlib
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
    
    # Tampilkan hasil di tkinter
    canvas = FigureCanvasTkAgg(fig, master=frame_results)
    canvas.get_tk_widget().pack(pady=10, padx=10, expand=True)
    canvas.draw()
    
    # Tampilkan gambar dengan bounding box di frame baru
    display_bounding_box(img_with_box)

def display_bounding_box(img_with_box):
    for widget in frame_bounding_box.winfo_children():
        widget.destroy()  # Menghapus semua elemen lama di frame_bounding_box
    
    img_rgb = cv2.cvtColor(img_with_box, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
    
    lbl_bounding_box = tk.Label(frame_bounding_box, image=img_tk)
    lbl_bounding_box.image = img_tk
    lbl_bounding_box.pack()

    # Reset dan tampilkan YOLO label yang baru
    for widget in frame_bounding_box_info.winfo_children():
        widget.destroy()  # Menghapus label YOLO lama
    
    # Tentukan nilai YOLO berdasarkan status kematangan
    if status_kematangan == "Matang":
        initial_yolo_value = "1"
    elif status_kematangan == "Mentah":
        initial_yolo_value = "0"
    elif status_kematangan == "Setengah Matang":
        initial_yolo_value = "2"
    else:
        initial_yolo_value = "Unknown"
    
    # Format label YOLO dengan status kematangan
    yolo_label_with_status = f"{initial_yolo_value} {yolo_label[2:]}"  # Ganti nilai pertama dengan status kematangan
    
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

# Canvas untuk scroll
canvas = tk.Canvas(root)
scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

# Masukkan elemen GUI ke dalam scrollable_frame
frame_input = LabelFrame(scrollable_frame, text="Gambar Asli", padx=10, pady=10)
frame_input.pack(side=tk.TOP, padx=20, pady=20)

lbl_img = tk.Label(frame_input)
lbl_img.pack()

btn_load = tk.Button(frame_input, text="Masukkan Citra", command=open_image, width=20, height=2)
btn_load.pack(pady=10)

btn_process = tk.Button(frame_input, text="Proses Citra", command=process_image, width=20, height=2)
btn_process.pack(pady=10)

# Frame untuk Hasil Analisis
frame_results = LabelFrame(scrollable_frame, text="Hasil Analisis", padx=10, pady=10)
frame_results.pack(pady=10, fill=tk.BOTH, expand=True, side=tk.TOP, anchor='n')

# Frame untuk Bounding Box
frame_bounding_box = LabelFrame(scrollable_frame, text="Bounding Box", padx=10, pady=10)
frame_bounding_box.pack(pady=10, fill=tk.BOTH, expand=True, side=tk.TOP, anchor='n')

# Frame untuk YOLO Format Label
frame_bounding_box_info = LabelFrame(scrollable_frame, text="YOLO Format Label", padx=10, pady=10)
frame_bounding_box_info.pack(pady=10, side=tk.TOP, anchor='n')


canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Mainloop
root.mainloop()
