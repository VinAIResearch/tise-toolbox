import os
import tarfile
import zipfile


google_drive_paths = {
    "text_to_images_models/DAMSMencoders/bird.zip": "https://drive.google.com/uc?id=1n3bDJ6uXZMtbgMtKSYJ3uExo11lrgaNW",
    "text_to_images_models/pretrained_models/cub_attn_gan_plus_plus_released.pth": "https://drive.google.com/uc?id=1pPtNww0Ift1v118cvYe6Fht-y2yal1tR",
    "text_to_images_models/pretrained_models/cub_counter_model.pth": "https://drive.google.com/uc?id=1PHTLhItcxyGJ4hFzMCdo76QcCQwAW2BP",
    "text_to_images_models/DAMSMencoders/coco.zip": "https://drive.google.com/uc?id=1kXB7HgKdEk-u25MFS2roB9GDnYAiyXg5",
    "text_to_images_models/pretrained_models/coco_attn_gan_plus_plus_released.pth": "https://drive.google.com/uc?id=1KSRyPL5S9_I9tGxDTBLNUHPuXzZ55TCR",
    "image_realism/IS/bird/inception_finetuned_models.zip": "https://drive.google.com/uc?id=1N2NI6BZW_bKz96CvWTNKU8SAeKuAEtis",
    "counting_alignment/weights.zip": "https://drive.google.com/uc?id=1W7hXzD3KsmoKJQzlBBhWTN3-sW--njqG",
    "object_fidelity/weights.zip": "https://drive.google.com/uc?id=1h5mdtCtf9ADqfSAfiqyppYGqipYRvvRg",
    "semantic_object_accuracy/weights.zip": "https://drive.google.com/uc?id=1XzyEOnta_1u4oJ6ebBdEGTK5-hwjxwv7",
}

for download_filename in google_drive_paths:
    if not os.path.isfile(download_filename):
        gdrive_url = google_drive_paths[download_filename]
        try:
            from gdown import download as drive_download

            drive_download(gdrive_url, download_filename, quiet=False)
        except ModuleNotFoundError:
            print(
                "gdown module not found.", "pip install gdown or, manually download the checkpoint file:", gdrive_url
            )

    if not os.path.isfile(download_filename):
        print(download_filename, " not found, you may need to manually download this file.")

    file_ext = download_filename.split(".")[-1]
    base_dir = "/".join(download_filename.split("/")[:-1])

    if file_ext in ["zip", "tgz"]:
        print("Extracting:", download_filename)

    if file_ext == "zip":
        with zipfile.ZipFile(download_filename, "r") as zip_ref:
            zip_ref.extractall(base_dir)

    if file_ext == "tgz":
        with tarfile.open(download_filename, "r") as tar_ref:
            tar_ref.extractall(base_dir)
