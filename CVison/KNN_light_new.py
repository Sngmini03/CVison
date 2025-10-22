import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from collections import Counter
from tqdm import tqdm
import py7zr
import zipfile
import tarfile

# ==================== 설정 ====================
ROOT = r"C:\Users\seung\OneDrive\사진\바탕 화면\ComputerVIsion\CVison\dataset"

TRAIN_ARCHIVE = os.path.join(ROOT, "train.7z")
TEST_ARCHIVE = os.path.join(ROOT, "test.7z")
LABEL_CSV = os.path.join(ROOT, "trainLabels.csv")

TRAIN_DIR = os.path.join(ROOT, "train", "train")
TEST_DIR = os.path.join(ROOT, "test","test", "test")

USE_HOG = True
IMG_SIZE = (32, 32)
K_VALUES = [1, 3, 5, 7, 9, 11, 15]
BEST_K = 5  # 초기값, 이후 교차검증으로 갱신됨

# ==================== 압축 해제 ====================
def extract_archive(archive_path: str, out_dir: str):
    """압축 파일 추출"""
    os.makedirs(out_dir, exist_ok=True)

    # 항상 재해제 시도 (특히 test가 덜 풀린 경우 대비)
    if any(os.scandir(out_dir)):
        print(f"[info] Re-extracting archive: {archive_path}")
    else:
        print(f"[info] Extracting: {archive_path} -> {out_dir}")

    if not os.path.exists(archive_path):
        print(f"[warn] Archive not found: {archive_path}")
        return

    lower = archive_path.lower()
    try:
        if lower.endswith(".7z"):
            with py7zr.SevenZipFile(archive_path, mode='r') as z:
                z.extractall(path=out_dir)
        elif lower.endswith(".zip"):
            with zipfile.ZipFile(archive_path, 'r') as z:
                z.extractall(out_dir)
        elif lower.endswith((".tar.gz", ".tgz", ".tar")):
            mode = "r:gz" if lower.endswith((".tar.gz", ".tgz")) else "r:"
            with tarfile.open(archive_path, mode) as tar:
                tar.extractall(out_dir)
        else:
            raise ValueError("Unsupported archive format")
    except Exception as e:
        print(f"[error] Failed to extract {archive_path}: {e}")
    print(f"[ok] Extracted to: {out_dir}")

# ==================== 이미지 처리 ====================
EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

def find_image_by_id(folder, _id):
    """ID로 이미지 파일 찾기"""
    base = str(_id)
    for ext in EXTS:
        p = os.path.join(folder, base + ext)
        if os.path.exists(p):
            return p
        p2 = os.path.join(folder, base.zfill(6) + ext)
        if os.path.exists(p2):
            return p2
    return None

def load_image_gray(path, size=(32, 32)):
    """이미지를 그레이스케일로 로드 (한글 경로 지원)"""
    try:
        img_array = np.fromfile(path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        return img
    except Exception as e:
        raise ValueError(f"Failed to read: {path} ({e})")

def extract_features(img_gray, use_hog=True):
    """특징 추출 (raw pixel + HOG)"""
    raw = (img_gray.flatten().astype(np.float32) / 255.0)
    if not use_hog:
        return raw
    h = hog(
        img_gray, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), block_norm="L2-Hys",
        transform_sqrt=True, feature_vector=True
    ).astype(np.float32)
    return np.concatenate([raw, h], axis=0)

# ==================== 학습 데이터 구축 (캐싱 + 체크포인트) ====================
def build_train(label_csv, img_dir, size=(32, 32), use_hog=True):
    cache_prefix = f"train_cache_{'hog' if use_hog else 'raw'}_{size[0]}x{size[1]}"
    X_cache = os.path.join(ROOT, cache_prefix + "_X.npy")
    y_cache = os.path.join(ROOT, cache_prefix + "_y.npy")
    id_cache = os.path.join(ROOT, cache_prefix + "_ids.npy")

    if all(os.path.exists(p) for p in [X_cache, y_cache, id_cache]):
        print(f"[cache] Loading cached train data from {ROOT}")
        return np.load(X_cache), np.load(y_cache), np.load(id_cache)

    print("[info] Building train features from scratch...")
    df = pd.read_csv(label_csv)
    X, y, ids, missing = [], [], [], []

    for idx, (_id, lab) in enumerate(tqdm(zip(df["id"], df["label"]), total=len(df), desc="Loading train images")):
        p = find_image_by_id(img_dir, _id)
        if p is None:
            missing.append(_id)
            continue
        try:
            g = load_image_gray(p, size=size)
            feat = extract_features(g, use_hog=use_hog)
            X.append(feat)
            y.append(lab)
            ids.append(_id)
        except Exception as e:
            print(f"[warn] Skipping {_id}: {e}")

        # 중간 저장 (5000장 단위)
        if (idx + 1) % 5000 == 0:
            np.save(X_cache, np.vstack(X).astype(np.float32))
            np.save(y_cache, np.array(y))
            np.save(id_cache, np.array(ids))
            print(f"[checkpoint] Saved partial cache ({len(X)} samples)")

    if missing:
        print(f"[warn] Missing {len(missing)} images")

    X = np.vstack(X).astype(np.float32)
    y = np.array(y)
    ids = np.array(ids)

    np.save(X_cache, X)
    np.save(y_cache, y)
    np.save(id_cache, ids)
    print(f"[cache] Saved full training cache to {ROOT}")
    return X, y, ids

# ==================== 테스트 데이터 구축 (캐싱 + 체크포인트) ====================
def build_test(img_dir, size=(32, 32), use_hog=True):
    cache_prefix = f"test_cache_{'hog' if use_hog else 'raw'}_{size[0]}x{size[1]}"
    X_cache = os.path.join(ROOT, cache_prefix + "_X.npy")
    id_cache = os.path.join(ROOT, cache_prefix + "_ids.npy")

    if os.path.exists(X_cache) and os.path.exists(id_cache):
        print(f"[cache] Loading cached test data from {ROOT}")
        return np.load(X_cache), np.load(id_cache)

    print("[info] Building test features from scratch...")
    paths = [p.path for p in os.scandir(img_dir) if p.is_file()]
    test_ids = []
    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]
        try:
            test_ids.append(int(base))
        except:
            continue

    test_ids = sorted(set(test_ids))
    X, ids_found = [], []

    for idx, _id in enumerate(tqdm(test_ids, desc="Loading test images")):
        fp = find_image_by_id(img_dir, _id)
        if fp is None:
            continue
        try:
            g = load_image_gray(fp, size=size)
            feat = extract_features(g, use_hog=use_hog)
            X.append(feat)
            ids_found.append(_id)
        except Exception as e:
            print(f"[warn] Skipping {_id}: {e}")

        # 중간 저장
        if (idx + 1) % 2000 == 0:
            np.save(X_cache, np.vstack(X).astype(np.float32))
            np.save(id_cache, np.array(ids_found))
            print(f"[checkpoint] Saved partial test cache ({len(X)} samples)")

    X = np.vstack(X).astype(np.float32)
    ids_found = np.array(ids_found)
    np.save(X_cache, X)
    np.save(id_cache, ids_found)
    print(f"[cache] Saved full test cache to {ROOT}")
    return X, ids_found

# ==================== K-NN 구현 ====================
class KNearestNeighbor:
    def __init__(self, k=1):
        self.k = k

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X, show_progress=False):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)
        iterator = tqdm(range(num_test), desc=f"Predicting (k={self.k})") if show_progress else range(num_test)
        for i in iterator:
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            k_nearest_indices = np.argpartition(distances, self.k - 1)[:self.k]
            k_nearest_labels = self.ytr[k_nearest_indices]
            Ypred[i] = Counter(k_nearest_labels).most_common(1)[0][0]
        return Ypred

# ==================== 평가 함수 ====================
def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1}

# ==================== 메인 실행 ====================
def main():
    # print("=== Step 1: Extracting Archives ===")
    # extract_archive(TRAIN_ARCHIVE, TRAIN_DIR)
    # extract_archive(TEST_ARCHIVE, TEST_DIR)

    print("\n=== Step 2: Loading Train Data ===")
    X_all, y_all, id_all = build_train(LABEL_CSV, TRAIN_DIR, size=IMG_SIZE, use_hog=USE_HOG)
    print(f"[ok] Train data loaded: {X_all.shape[0]} samples, {X_all.shape[1]} features")

    le = LabelEncoder()
    y_all_enc = le.fit_transform(y_all)

    print("\n=== Step 3: Train/Validation Split ===")
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all_enc, test_size=0.2, random_state=42, stratify=y_all_enc
    )
    print(f"[ok] Train: {len(X_train)}, Validation: {len(X_val)}")

    print("\n=== Step 4: Loading Test Data ===")
    X_test_real, test_ids = build_test(TEST_DIR, size=IMG_SIZE, use_hog=USE_HOG)
    print(f"[ok] Test: {X_test_real.shape[0]} samples")

    print("\n=== Step 5: Hyperparameter Selection (Validation Set) ===")
    val_results = {}
    for k in K_VALUES:
        knn = KNearestNeighbor(k=k)
        knn.train(X_train, y_train)
        val_pred = knn.predict(X_val, show_progress=True)
        metrics = evaluate_model(y_val, val_pred)
        val_results[k] = metrics
        print(f"k={k}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")

    best_k = max(val_results, key=lambda k: val_results[k]['accuracy'])
    print(f"\n[info] Best k from validation: {best_k}")

    print("\n=== Step 6: Final Training & Prediction on Test Set ===")
    knn_final = KNearestNeighbor(k=best_k)
    knn_final.train(X_all, y_all_enc)
    test_pred_enc = knn_final.predict(X_test_real, show_progress=True)
    test_pred_lbl = le.inverse_transform(test_pred_enc)

    submission = pd.DataFrame({"id": test_ids, "label": test_pred_lbl}).sort_values("id")
    save_path = os.path.join(ROOT, "submission.csv")
    submission.to_csv(save_path, index=False)
    print(f"[info] Submission saved to {save_path}")
    print(submission.head())

if __name__ == "__main__":
    main()
