import os
import tarfile
import zipfile


google_drive_paths = {
    "text_to_images_models/data/bird.zip": "https://drive.google.com/uc?id=1dZFKKGiEHU5mZV7hbTQmwS4MnhxM3vWG",
    "text_to_images_models/data/birds/CUB_200_2011.tgz": "https://drive.google.com/uc?id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45",
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
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar_ref, base_dir)
