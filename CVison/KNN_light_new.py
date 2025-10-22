import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from collections import Counter
from tqdm import tqdm

# ==================== 설정 ====================
ROOT = r"C:\Users\seung\OneDrive\사진\바탕 화면\ComputerVIsion\CVison\dataset"

LABEL_CSV = os.path.join(ROOT, "trainLabels.csv")
TRAIN_DIR = os.path.join(ROOT, "train", "train")

USE_HOG = True
IMG_SIZE = (32, 32)
K_VALUES = [1, 3, 5, 7, 9, 11, 15, 20]  # validation용 후보 k
BEST_K = 5  # 초기값

# ==================== 이미지 처리 ====================
EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

def find_image_by_id(folder, _id):
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
    img_array = np.fromfile(path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img

def extract_features(img_gray, use_hog=True):
    raw = (img_gray.flatten().astype(np.float32) / 255.0)
    if not use_hog:
        return raw
    h = hog(
        img_gray, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), block_norm="L2-Hys", 
        transform_sqrt=True, feature_vector=True
    ).astype(np.float32)
    return np.concatenate([raw, h], axis=0)

# ==================== 학습 데이터 로드 (캐시 사용) ====================
def build_train(label_csv, img_dir, size=(32, 32), use_hog=True):
    cache_prefix = f"train_cache_{'hog' if use_hog else 'raw'}_{size[0]}x{size[1]}"
    X_cache = os.path.join(ROOT, cache_prefix + "_X.npy")
    y_cache = os.path.join(ROOT, cache_prefix + "_y.npy")
    id_cache = os.path.join(ROOT, cache_prefix + "_ids.npy")

    if os.path.exists(X_cache) and os.path.exists(y_cache) and os.path.exists(id_cache):
        print(f"[cache] Loading cached train data from {ROOT}")
        X = np.load(X_cache)
        y = np.load(y_cache)
        ids = np.load(id_cache)
        print(f"[ok] Cached data loaded: {X.shape[0]} samples")
        return X, y, ids
    else:
        raise FileNotFoundError("Cache not found. Please ensure cached training data exists.")

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
            k_nearest_indices = np.argpartition(distances, self.k-1)[:self.k]
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

# ==================== 5-Fold Cross Validation ====================
def cross_validate_knn(X, y, k_values, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {k: [] for k in k_values}

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\n=== Fold {fold_idx}/{n_folds} ===")
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        for k in k_values:
            knn = KNearestNeighbor(k=k)
            knn.train(X_train_fold, y_train_fold)
            val_pred = knn.predict(X_val_fold)
            metrics = evaluate_model(y_val_fold, val_pred)
            results[k].append(metrics)
            print(f"k={k}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
    return results

# ==================== 메인 ====================
def main():
    print("=== Step 1: Loading Cached Train Data ===")
    X_all, y_all, id_all = build_train(LABEL_CSV, TRAIN_DIR, size=IMG_SIZE, use_hog=USE_HOG)
    print(f"Loaded {X_all.shape[0]} samples, feature dim: {X_all.shape[1]}")

    # 라벨 인코딩
    le = LabelEncoder()
    y_all_enc = le.fit_transform(y_all)

    # ==================== Train / Val / Test Split ====================
    print("\n=== Step 2: Splitting Dataset (Train/Val/Test) ===")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all_enc, test_size=0.3, random_state=42, stratify=y_all_enc
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ==================== Validation으로 Best k 선택 ====================
    print("\n=== Step 3: Hyperparameter Selection (Validation Set) ===")
    val_results = {}
    for k in K_VALUES:
        knn = KNearestNeighbor(k=k)
        knn.train(X_train, y_train)
        val_pred = knn.predict(X_val, show_progress=False)
        metrics = evaluate_model(y_val, val_pred)
        val_results[k] = metrics
        print(f"k={k}: Acc={metrics['accuracy']:.4f}, Prec={metrics['precision']:.4f}, "
              f"Rec={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}")

    best_k = max(val_results, key=lambda k: val_results[k]['accuracy'])
    print(f"\n[info] Best k from validation: {best_k}")

    # ==================== Cross Validation (선택사항) ====================
    print("\n=== Step 4: 5-Fold Cross Validation (Train set only) ===")
    cv_results = cross_validate_knn(X_train, y_train, K_VALUES)

    # ==================== 최종 평가 ====================
    print(f"\n=== Step 5: Final Evaluation (Test Set, k={best_k}) ===")
    knn_final = KNearestNeighbor(k=best_k)
    knn_final.train(X_train, y_train)
    test_pred = knn_final.predict(X_test, show_progress=True)
    final_metrics = evaluate_model(y_test, test_pred)

    print("\nFinal Test Metrics:")
    print(f"Accuracy : {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall   : {final_metrics['recall']:.4f}")
    print(f"F1-Score : {final_metrics['f1_score']:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, test_pred, target_names=le.classes_))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_pred))

if __name__ == "__main__":
    main()
