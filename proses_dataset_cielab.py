import cv2
import numpy as np
import os

# Fungsi untuk gamma correction
def adjust_gamma(img, gamma=1.0):
    mean_intensity = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    target_intensity = 128
    gamma = target_intensity / mean_intensity
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

# Fungsi untuk memproses gambar
def process_image(image_path, output_base_path):
    # Load gambar
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Tidak dapat membaca gambar {image_path}")
        return
    
    img = cv2.resize(img, (250, 300))
    img_gamma = adjust_gamma(img)  # gamma correction
    
    # Konversi ke LAB dan masking warna merah
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    _, a_channel, _ = cv2.split(img_lab)
    lower_red, upper_red = 140, 200
    mask_red = cv2.inRange(a_channel, lower_red, upper_red)
    
    # Deteksi tepi dan kontur
    gray = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = sum(cv2.contourArea(cnt) for cnt in contours)
    
    # Final masking area merah
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_red_cleaned = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_red_area = cv2.bitwise_and(img_gamma, img_gamma, mask=mask_red_cleaned)
    
    # Hitung area merah
    red_area = cv2.countNonZero(mask_red_cleaned)
    if total_area > 0:
        maturity_persentase = (red_area / total_area) * 100
    else:
        maturity_persentase = 0.0
    
    # Klasifikasi kematangan
    if maturity_persentase >= 85:
        status_kematangan = "Matang"
    elif 20 < maturity_persentase < 85:
        status_kematangan = "Setengah Matang"
    else:
        status_kematangan = "Mentah"
        
    # Tampilkan hasil
    print(f"\n----    Hasil untuk {os.path.basename(image_path)}    ----")
    print(f"Total luas: {total_area:.2f} piksel")
    print(f"Luas area merah: {red_area:.2f} piksel")
    print(f"Persentase kematangan: {maturity_persentase:.2f}%")
    print(f"Status kematangan: {status_kematangan}")
    
    # Buat folder output berdasarkan klasifikasi
    klasifikasi_folder = os.path.join(output_base_path, status_kematangan)
    if not os.path.exists(klasifikasi_folder):
        os.makedirs(klasifikasi_folder)
    
    # Simpan gambar hasil olahan
    output_path = os.path.join(klasifikasi_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, img)  # Simpan gambar asli ke folder klasifikasi

# Folder input dan output
input_folder = r"D:\Materi Kuliah Debby\Project Semester 5\RoboBloom\dataset strawberry"
output_folder = os.path.join(input_folder, "output")  # Folder output dibuat di dalam folder input

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Proses semua gambar di folder input
for file_name in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file_name)
    if os.path.isfile(file_path) and file_name.lower().endswith((".jpg", ".png", ".jpeg")):
        process_image(file_path, output_folder)

print("\nSemua gambar telah diproses dan disimpan di folder output.")
