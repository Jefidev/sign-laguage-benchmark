from lsfb_dataset import Downloader
import time
from requests.exceptions import SSLError


downloader = Downloader(
    dataset="cont",
    destination="/gpfs/projects/acad/lsfb/datasets/lsfb_v2/cont",
    include_videos=True,
    include_raw_poses=True,
    include_cleaned_poses=True,
    skip_existing_files=True,
)
finished = False

while finished == False:
    try:
        downloader.download()
        finished = True
    except Exception as e:
        print("SSL Error, retrying in 20 sec...")
        print(e)
        time.sleep(20)
