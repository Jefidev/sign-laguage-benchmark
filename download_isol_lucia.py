from lsfb_dataset import Downloader
import time
from requests.exceptions import SSLError

finished = False

downloader = Downloader(
    dataset="isol",
    destination="/gpfs/projects/acad/lsfb/datasets/lsfb_v2/isol",
    include_videos=True,
    include_raw_poses=True,
    include_cleaned_poses=True,
    skip_existing_files=True,
)

while finished == False:
    try:
        downloader.download()
        finished = True
    except Exception as e:
        print("SSL Error, retrying in 20 sec...")
        time.sleep(20)
