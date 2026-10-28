
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import feature, io, util
from skimage.feature import graycomatrix, graycoprops
from skimage.transform import resize

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
MAPPING = ROOT / "data" / "carinthia-s.csv"
CACHE = ROOT / "outputs" / "feature_cache" / "original_features.npz"


def read_csv_auto(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, sep=None, engine="python")


def stem(value):
    return Path(str(value).replace("\\", "/")).stem.lower()


def resolve_image(filepath):
    path = Path(str(filepath))
    candidates = [
        path,
        ROOT / path,
        ROOT / "data" / "images" / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Image not found: {filepath}")


def resolve_mask(filepath):
    path = Path(str(filepath))
    candidates = [
        path,
        ROOT / path,
        ROOT / "data" / "masks" / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Mask not found: {filepath}")


def load_manifest():
    manifest = read_csv_auto(MANIFEST)
    mapping = read_csv_auto(MAPPING)

    manifest.columns = manifest.columns.str.strip().str.lower()
    mapping.columns = mapping.columns.str.strip().str.lower()

    required_manifest = {"filepath", "class", "split"}
    required_mapping = {"image_path", "mask_path", "filename", "label"}

    if not required_manifest.issubset(manifest.columns):
        raise ValueError(
            f"manifest.csv requires {required_manifest}; "
            f"found {manifest.columns.tolist()}"
        )

    if not required_mapping.issubset(mapping.columns):
        raise ValueError(
            f"carinthia-s.csv requires {required_mapping}; "
            f"found {mapping.columns.tolist()}"
        )

    manifest = manifest.copy()
    mapping = mapping.copy()

    manifest["sample_id"] = manifest["filepath"].map(stem)
    mapping["sample_id"] = mapping["filename"].map(stem)

    manifest["label"] = pd.to_numeric(
        manifest["class"], errors="raise"
    ).astype(int)

    mapping["mapping_label"] = pd.to_numeric(
        mapping["label"], errors="raise"
    ).astype(int)

    manifest["split"] = (
        manifest["split"].astype(str).str.strip().str.lower()
    )

    if manifest["sample_id"].duplicated().any():
        raise ValueError("Duplicate image filenames in manifest.csv")

    if mapping["sample_id"].duplicated().any():
        raise ValueError("Duplicate image filenames in carinthia-s.csv")

    if not set(manifest["split"]).issubset({"train", "val", "test"}):
        raise ValueError("Split values must be train, val, or test.")

    data = manifest.merge(
        mapping[
            ["sample_id", "image_path", "mask_path", "mapping_label"]
        ],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    if data["mapping_label"].isna().any():
        raise ValueError(
            "Some manifest images could not be matched to carinthia-s.csv."
        )

    if (data["label"] != data["mapping_label"]).any():
        raise ValueError(
            "Labels disagree between manifest.csv and carinthia-s.csv."
        )

    if not set(data["label"]).issubset(set(range(1, 7))):
        raise ValueError("Expected class labels 1 through 6.")

    data["image_file"] = data["filepath"].map(resolve_image)
    data["mask_file"] = data["mask_path"].map(resolve_mask)

    return data


def extract_features(filepath):
    image = io.imread(str(filepath), as_gray=True)
    image = util.img_as_float32(image)

    image = resize(
        image,
        (128, 128),
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)

    image = np.clip(image, 0, 1)

    # HOG
    hog = feature.hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    # LBP
    gray = np.round(image * 255).astype(np.uint8)
    lbp = feature.local_binary_pattern(
        gray, P=24, R=3, method="uniform"
    )
    lbp_hist, _ = np.histogram(
        lbp.ravel(), bins=np.arange(27), range=(0, 26)
    )
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist /= max(lbp_hist.sum(), 1)

    # GLCM
    quantized = (gray // 8).astype(np.uint8)
    glcm = graycomatrix(
        quantized,
        distances=[1, 2, 4],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32,
        symmetric=True,
        normed=True,
    )

    texture = []
    for prop in [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM",
    ]:
        texture.extend(graycoprops(glcm, prop).ravel())

    return np.concatenate([
        hog,
        lbp_hist,
        np.asarray(texture, dtype=np.float32),
    ]).astype(np.float32)


def get_features(data, rebuild=False):
    CACHE.parent.mkdir(parents=True, exist_ok=True)

    # Store IDs as strings, not NumPy object arrays.
    ids = data["sample_id"].astype(str).to_numpy(dtype=str)

    if CACHE.exists() and not rebuild:
        try:
            with np.load(CACHE, allow_pickle=False) as cached:
                cached_ids = cached["sample_ids"].astype(str)

                if np.array_equal(cached_ids, ids):
                    print("Loading cached features...")
                    return cached["X"].astype(np.float32)

                print("Feature cache IDs differ; rebuilding cache.")

        except (ValueError, KeyError, OSError) as exc:
            print(f"Feature cache cannot be reused ({exc}); rebuilding.")

    features_list = []

    for i, path in enumerate(data["image_file"], start=1):
        if i % 100 == 0 or i == len(data):
            print(f"Extracting features: {i}/{len(data)}")
        features_list.append(extract_features(path))

    X = np.vstack(features_list).astype(np.float32)

    np.savez_compressed(
        CACHE,
        X=X,
        sample_ids=ids,
    )

    print(f"Saved features: {X.shape}")
    return X
