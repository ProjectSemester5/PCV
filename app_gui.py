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

def process_image():
    global img_path, img_gamma, status_kematangan, maturity_persentase
    
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
    
    # Tampilkan hasil
    display_results(img, img_gamma, mask_red, edges, cleaned_mask, contours)

def display_results(img, img_gamma, mask_red, edges, cleaned_mask, contours):
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
    canvas.get_tk_widget().pack()
    canvas.draw()

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
root.geometry("900x700")

# Frame untuk gambar asli
frame_input = LabelFrame(root, text="Gambar Asli", padx=10, pady=10)
frame_input.pack(side=tk.LEFT, padx=20, pady=20)

lbl_img = tk.Label(frame_input)
lbl_img.pack()

btn_load = tk.Button(frame_input, text="Masukkan Citra", command=open_image, width=20, height=2)
btn_load.pack(pady=10)

btn_process = tk.Button(frame_input, text="Proses Citra", command=process_image, width=20, height=2)
btn_process.pack(pady=10)

# Frame untuk hasil analisis
frame_results = LabelFrame(root, text="Hasil Analisis", padx=10, pady=10)
frame_results.pack(side=tk.RIGHT, padx=20, pady=20, fill=tk.BOTH, expand=True)

# Mainloop
root.mainloop()