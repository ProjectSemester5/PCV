import cv2
import numpy as np
import os

# Fungsi untuk melakukan gamma correction
def adjust_lightness(img_lab, gamma=1.0):
    l_channel, a_channel, b_channel = cv2.split(img_lab)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    l_channel_adjusted = cv2.LUT(l_channel, table)
    img_lab_adjusted = cv2.merge((l_channel_adjusted, a_channel, b_channel))
    return img_lab_adjusted

# Fungsi untuk menggambar bounding box
def draw_bounding_box(img, contours):
    if not contours:
        return img, None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    img_with_box = img.copy()
    cv2.rectangle(img_with_box, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img_with_box, (x, y, w, h)

# Fungsi untuk menghitung format YOLO
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

# Fungsi utama untuk memproses gambar
def process_images(image_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(image_folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(image_folder, filename)
            print(f"Memproses {img_path}...")

            img = cv2.imread(img_path)
            img = cv2.resize(img, (300, 300))

            # Konversi ke LAB color space
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            img_lab_adjusted = adjust_lightness(img_lab, gamma=1.0)
            img_gamma = cv2.cvtColor(img_lab_adjusted, cv2.COLOR_LAB2BGR)

            # Masking merah dan kuning
            mask_red = cv2.inRange(img_lab[:, :, 1], 140, 210)  # 'a' channel (red)
            mask_yellow = cv2.inRange(img_lab[:, :, 2], 165, 200)  # 'b' channel (yellow)
            combined_mask = cv2.bitwise_or(mask_red, mask_yellow)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

            # Deteksi kontur dan area
            contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total_area = sum(cv2.contourArea(cnt) for cnt in contours)
            red_area = cv2.countNonZero(mask_red)

            # Hitung persentase kematangan
            maturity_persentase = (red_area / total_area) * 100 if total_area > 0 else 0.0

            # Klasifikasi kematangan
            if maturity_persentase >= 80:
                status_kematangan = "Matang"
            elif 20 < maturity_persentase < 80:
                status_kematangan = "Setengah Matang"
            else:
                status_kematangan = "Mentah"

            # Gambar bounding box dan hitung YOLO format label
            img_with_box, bbox = draw_bounding_box(img, contours)
            yolo_label = calculate_yolo_format(img.shape, bbox)
            
            # Sesuaikan label berdasarkan status kematangan
            initial_yolo_value = {"Matang": "1", "Mentah": "0", "Setengah Matang": "2"}.get(status_kematangan, "Unknown")
            yolo_label_with_status = f"{initial_yolo_value} {yolo_label[2:]}"

            # Simpan label YOLO ke file teks
            label_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
            with open(label_path, "w") as file:
                file.write(yolo_label_with_status)

            print(f"Label YOLO disimpan di {label_path} (Kematangan: {status_kematangan}, {maturity_persentase:.2f}%)")

# Jalankan pemrosesan
if __name__ == "__main__":
    # path img
    image_folder = r"D:\Kuliah\Semester_5\Project\yolo_strawberry_train\test\images"
    output_folder = "labels"      
    process_images(image_folder, output_folder)
        
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    process_images(image_folder, output_folder)
