import os
import tarfile
import zipfile


google_drive_paths = {
    "classifier_calibration/data/coco_object_validation_feature_data.npz": "https://drive.google.com/uc?id=1btKm82ImFYa63lM88pcGxdla6inuaYbB",
    "classifier_calibration/data/cub_validation_feature_data.npz": "https://drive.google.com/uc?id=1PAQl2K4Ul33jFiHisQL_Iub3DGug2wnk",
    "classifier_calibration/data/image_net_validation_feature_data.npz": "https://drive.google.com/uc?id=1fY-84uIbD2--j_bRd0pjhupPPW2oD4Qx",
    "classifier_calibration/data/tf_image_net_validation_feature_data.npz": "https://drive.google.com/uc?id=1MaCE-UmUV1XsRRj3A71mE-pc2Q0huHLT",
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
