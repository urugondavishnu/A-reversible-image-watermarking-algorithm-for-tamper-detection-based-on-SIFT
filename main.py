import cv2
import numpy as np
import pywt
import random
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

# ======================= Utility Functions ==========================
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"{path} not found.")
    return cv2.resize(img, (512, 512))

def generate_watermark(image, bits=8):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten().astype(int)
    return ''.join(format(x % 2, 'b') for x in hist[:bits])

def get_sift_points(image, num_points=8):
    sift = cv2.SIFT_create()
    keypoints = sift.detect(image, None)
    keypoints = sorted(keypoints, key=lambda x: -x.response)
    return [(int(kp.pt[1]), int(kp.pt[0])) for kp in keypoints[:num_points]]

# ======================= High-PSNR Reversible Watermarking ==========================
def reversible_embed(image, watermark, alpha=0.001):
    coeffs = pywt.wavedec2(image, 'haar', level=2)
    LL2, (LH2, HL2, HH2), (LH1, HL1, HH1) = coeffs
    LH1_backup = LH1.copy()
    U, S, Vt = np.linalg.svd(LH1, full_matrices=False)
    S_embed = S.copy()
    for i in range(min(len(watermark), 8)):
        idx = -i-1
        if (int(S[idx] / S[0]) % 2) != int(watermark[i]):
            S_embed[idx] += alpha if watermark[i] == '1' else -alpha
    LH1_mod = np.dot(U, np.dot(np.diag(S_embed), Vt))
    coeffs_modified = (LL2, (LH2, HL2, HH2), (LH1_mod, HL1, HH1))
    watermarked = pywt.waverec2(coeffs_modified, 'haar')
    return np.clip(watermarked, 0, 255).astype(np.uint8), LH1_backup

def reversible_extract(image, LH1_backup):
    coeffs = pywt.wavedec2(image, 'haar', level=2)
    LL2, (LH2, HL2, HH2), (_, HL1, HH1) = coeffs
    coeffs_restored = (LL2, (LH2, HL2, HH2), (LH1_backup, HL1, HH1))
    restored = pywt.waverec2(coeffs_restored, 'haar')
    return np.clip(restored, 0, 255).astype(np.uint8)

# ======================= Tampering & Attack Simulation ==========================
def tamper_image_random(image, size=30, count=5):
    tampered = image.copy()
    h, w = image.shape
    for _ in range(count):
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        tampered[y:y+size, x:x+size] = 255 - tampered[y:y+size, x:x+size]
    return tampered

def apply_attack(image, kind):
    if kind == "gaussian_blur":
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif kind == "median":
        return cv2.medianBlur(image, 3)
    elif kind == "gaussian_noise":
        noise = np.random.normal(0, 10, image.shape)
        return np.clip(image + noise, 0, 255).astype(np.uint8)
    elif kind == "salt_pepper":
        prob = 0.03
        noisy = image.copy()
        rnd = np.random.rand(*image.shape)
        noisy[rnd < prob] = 0
        noisy[rnd > 1 - prob] = 255
        return noisy
    elif kind == "jpeg20":
        _, encimg = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 20])
        return cv2.imdecode(encimg, 0)
    elif kind == "jpeg50":
        _, encimg = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        return cv2.imdecode(encimg, 0)
    elif kind == "scaling":
        small = cv2.resize(image, (460, 460))
        return cv2.resize(small, (512, 512))
    elif kind == "rotation":
        M = cv2.getRotationMatrix2D((256, 256), 30, 1)
        rotated = cv2.warpAffine(image, M, (512, 512))
        M_inv = cv2.getRotationMatrix2D((256, 256), -30, 1)
        return cv2.warpAffine(rotated, M_inv, (512, 512))
    elif kind == "shearing":
        rows, cols = image.shape
        M = np.float32([[1, 0.25, 0], [0, 1, 0]])
        return cv2.warpAffine(image, M, (cols, rows))
    return image

def detect_tampering(original, tampered, threshold=25):
    diff = cv2.absdiff(original, tampered)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return mask

# ======================= Evaluation ==========================
def evaluate_metrics(original, modified):
    return psnr(original, modified), ssim(original, modified)

def normalized_correlation(img1, img2):
    img1 = img1.astype(np.float32).flatten()
    img2 = img2.astype(np.float32).flatten()
    return np.dot(img1, img2) / (np.linalg.norm(img1) * np.linalg.norm(img2))

def visualize_all(original, watermarked, tampered, mask, restored):
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 3, 1), plt.imshow(original, cmap='gray'), plt.title("Original"), plt.axis('off')
    plt.subplot(2, 3, 2), plt.imshow(watermarked, cmap='gray'), plt.title("Watermarked"), plt.axis('off')
    plt.subplot(2, 3, 3), plt.imshow(tampered, cmap='gray'), plt.title("Tampered"), plt.axis('off')
    plt.subplot(2, 3, 4), plt.imshow(mask, cmap='hot'), plt.title("Tamper Heatmap"), plt.axis('off')
    plt.subplot(2, 3, 5), plt.imshow(restored, cmap='gray'), plt.title("Restored (Reversible)"), plt.axis('off')
    plt.tight_layout()
    plt.show()

# ======================= Main Execution ==========================
def main():
    path = input("Enter grayscale image path (e.g., lena.png, baboon.png): ").strip()
    image = load_image(path)

    print(">> Detecting SIFT keypoints...")
    sift_points = get_sift_points(image)
    print(f">> Top SIFT locations: {sift_points}")

    watermark = generate_watermark(image, bits=8)
    print(">> Embedding watermark with reversible DWT-SVD (level=2)...")
    watermarked, LH1_backup = reversible_embed(image, watermark)

    print(">> Reversibly extracting image using LH1 backup...")
    restored = reversible_extract(watermarked, LH1_backup)

    tampered = tamper_image_random(watermarked)
    tamper_mask = detect_tampering(watermarked, tampered)

    print("\n>> Robustness NC (Normalized Correlation) under attacks:")
    for atk in ["gaussian_blur", "salt_pepper", "jpeg20", "scaling", "rotation"]:
        attacked = apply_attack(watermarked, atk)
        nc = normalized_correlation(image, attacked)
        print(f"{atk:<15} NC: {nc:.4f}")

    psnr_val, ssim_val = evaluate_metrics(image, watermarked)
    print(f"\nPSNR (Watermarked): {psnr_val:.2f} dB")
    print(f"SSIM (Watermarked): {ssim_val:.4f}")

    visualize_all(image, watermarked, tampered, tamper_mask, restored)

if _name_ == "_main_":
    main()