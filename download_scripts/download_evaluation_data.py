import os
import tarfile
import zipfile


google_drive_paths = {
    "image_realism/captions/cub_image_realism_captions.pkl": "https://drive.google.com/uc?id=1CiVX30GYstnDTeIALzcmIiZwEOB_8GmV",
    "text_relevance/captions/CUB_RP_captions.pkl": "https://drive.google.com/uc?id=1rgviKILsxaZC7geYAzQG_LtNs9ZiTK4n",
    "counting_alignment/captions/CA_input_captions.pkl": "https://drive.google.com/uc?id=102oFdZSLLeYVuMDH4yDdta8ZqTGyCw9F",
    "image_realism/captions/coco_image_realism_captions.pkl": "https://drive.google.com/uc?id=1dlJio9C1ALkq8HeSpv4eN5qKkouCSsWj",
    "image_realism/FID/data.zip": "https://drive.google.com/uc?id=1b-xxEHQDFqzYVKanmHJNRgRuxOAQ0k-D",
    "object_fidelity/O-FID/data.zip": "https://drive.google.com/uc?id=18dbKvDADfs1psx3JtVylDE7NR0YAZzz5",
    "positional_alignment/captions/PA_input_captions.pkl": "https://drive.google.com/uc?id=1GIvnWUblvKtyb2KSngSbYeTE8qtxwyZi",
    "semantic_object_accuracy/captions.zip": "https://drive.google.com/uc?id=17GHoILW9KTwhh2aFbCNcC4zg1RPXAFnk",
    "text_relevance/captions/COCO_RP_captions.pkl": "https://drive.google.com/uc?id=1bZvXauK5443dFnVeFL8bIdxG8IyaZiXY",
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
