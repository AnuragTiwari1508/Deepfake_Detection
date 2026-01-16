import os
import glob
import random
import shutil
import yaml
import cv2
import numpy as np

from tqdm import tqdm

from preprocessing.face_detect import FaceDetector


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def collect_videos(base_dir):
    video_exts = ["*.mp4", "*.avi", "*.mov", "*.mkv"]

    orig_root = os.path.join(base_dir, "DFD_original sequences")
    manip_root = os.path.join(base_dir, "DFD_manipulated_sequences")

    orig_videos = []
    manip_videos = []

    for ext in video_exts:
        orig_videos.extend(
            glob.glob(os.path.join(orig_root, "**", ext), recursive=True)
        )
        manip_videos.extend(
            glob.glob(os.path.join(manip_root, "**", ext), recursive=True)
        )

    return orig_videos, manip_videos


def ensure_clean_dir(path):
    os.makedirs(path, exist_ok=True)
    for f in glob.glob(os.path.join(path, "*")):
        if os.path.isfile(f):
            os.remove(f)


def extract_faces_from_videos(video_paths, label_name, out_dir, face_detector, fps, max_faces=None):
    os.makedirs(out_dir, exist_ok=True)
    saved_paths = []

    for video_path in tqdm(video_paths, desc=f"Processing {label_name} videos"):
        faces = []
        try:
            faces = face_detector.process_video(video_path, fps=fps)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue

        base = os.path.splitext(os.path.basename(video_path))[0]

        for idx, face_img in enumerate(faces):
            out_name = f"{base}_f{idx:05d}.png"
            out_path = os.path.join(out_dir, out_name)
            bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
            ok = cv2.imwrite(out_path, bgr)
            if ok:
                saved_paths.append(out_path)
                if max_faces is not None and len(saved_paths) >= max_faces:
                    return saved_paths

    return saved_paths


def build_balanced_dataset():
    config = load_config()
    fps = config["data"]["frame_fps"]

    orig_videos, manip_videos = collect_videos("data")

    print(f"Found {len(orig_videos)} original videos")
    print(f"Found {len(manip_videos)} manipulated videos")

    if len(orig_videos) == 0 and len(manip_videos) == 0:
        print("No DFD videos found under data/.")
        return

    face_detector = FaceDetector()

    raw_real_dir = os.path.join("data", "dfd_faces_real_all")
    raw_fake_dir = os.path.join("data", "dfd_faces_fake_all")
    os.makedirs(raw_real_dir, exist_ok=True)
    os.makedirs(raw_fake_dir, exist_ok=True)

    existing_real = glob.glob(os.path.join(raw_real_dir, "*.png"))
    existing_fake = glob.glob(os.path.join(raw_fake_dir, "*.png"))

    if existing_real and existing_fake:
        real_paths = existing_real
        fake_paths = existing_fake
        print(f"Using existing face crops: {len(real_paths)} real, {len(fake_paths)} fake")
    else:
        ensure_clean_dir(raw_real_dir)
        ensure_clean_dir(raw_fake_dir)
        real_paths = extract_faces_from_videos(
            orig_videos, "real", raw_real_dir, face_detector, fps
        )
        fake_paths = extract_faces_from_videos(
            manip_videos, "fake", raw_fake_dir, face_detector, fps, max_faces=len(real_paths)
        )

    print(f"Extracted {len(real_paths)} real face crops")
    print(f"Extracted {len(fake_paths)} fake face crops")

    if len(real_paths) == 0 or len(fake_paths) == 0:
        print("Not enough faces in one of the classes to build a balanced dataset.")
        return

    min_count = min(len(real_paths), len(fake_paths))
    print(f"Balancing to {min_count} samples per class")

    random.shuffle(real_paths)
    random.shuffle(fake_paths)

    real_paths = real_paths[:min_count]
    fake_paths = fake_paths[:min_count]

    real_target = os.path.join("data", "real")
    fake_target = os.path.join("data", "fake")

    ensure_clean_dir(real_target)
    ensure_clean_dir(fake_target)

    for src in real_paths:
        dst = os.path.join(real_target, os.path.basename(src))
        shutil.copy2(src, dst)

    for src in fake_paths:
        dst = os.path.join(fake_target, os.path.basename(src))
        shutil.copy2(src, dst)

    print(f"Final balanced dataset:")
    print(f"  Real: {len(real_paths)} images -> {real_target}")
    print(f"  Fake: {len(fake_paths)} images -> {fake_target}")


if __name__ == "__main__":
    build_balanced_dataset()
