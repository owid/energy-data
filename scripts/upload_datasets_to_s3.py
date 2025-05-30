"""Upload OWID energy dataset files to S3 and make them publicly available.

This script requires OWID credentials to write files in the S3 bucket.

Files should be accessible at the following urls:
* https://nyc3.digitaloceanspaces.com/owid-public/data/energy/owid-energy-data.csv
* https://nyc3.digitaloceanspaces.com/owid-public/data/energy/owid-energy-data.xlsx
* https://nyc3.digitaloceanspaces.com/owid-public/data/energy/owid-energy-data.json

"""

import argparse
import os

from owid.datautils.s3 import S3
from tqdm.auto import tqdm

from shared import OUTPUT_DIR

# S3 bucket name and folder where energy dataset files will be stored.
S3_BUCKET_NAME = "owid-public"
S3_ENERGY_DIR = os.path.join("data", "energy")
# Local files to upload.
FILES_TO_UPLOAD = {
    os.path.join(OUTPUT_DIR, "owid-energy-data.csv"): os.path.join(
        S3_ENERGY_DIR, "owid-energy-data.csv"
    ),
    os.path.join(OUTPUT_DIR, "owid-energy-data.json"): os.path.join(
        S3_ENERGY_DIR, "owid-energy-data.json"
    ),
    os.path.join(OUTPUT_DIR, "owid-energy-data.xlsx"): os.path.join(
        S3_ENERGY_DIR, "owid-energy-data.xlsx"
    ),
}


def main(files_to_upload, s3_bucket_name=S3_BUCKET_NAME):
    # Make files publicly available.
    public = True
    # Initialise S3 client.
    s3 = S3()
    # Upload and make public each of the files.
    for local_file in tqdm(files_to_upload):
        s3_file = files_to_upload[local_file]
        tqdm.write(
            f"Uploading file {local_file} to S3 bucket {s3_bucket_name} as {s3_file}."
        )
        s3.upload_to_s3(
            local_path=local_file,
            s3_path=f"s3://{s3_bucket_name}/{s3_file}",
            public=public,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    main(files_to_upload=FILES_TO_UPLOAD)
