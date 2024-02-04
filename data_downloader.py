import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup

data_dir = "./data"
max_concurrent_downloads = 48

# URLs
scroll_one_base_url = ("scroll1", "http://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths/")
scroll_two_base_url = ("scroll2", "http://dl.ash2txt.org/full-scrolls/Scroll2.volpkg/paths/")

urls = [scroll_one_base_url, scroll_two_base_url]

username = "registeredusers"
password = "only"


def get_url_data_path(data_url):
    return os.path.join(data_dir, data_url[0])


def saved_segments(base_data_dir, data_urls):
    data_url_results = {}
    for data_url in data_urls:
        saved_dir = os.path.join(base_data_dir, data_url[0])
        if os.path.exists(saved_dir) and os.path.isdir(saved_dir):
            exclude_list = ['.', '..', '.DS_Store', 'Thumbs.db', 'desktop.ini', '.Trashes', 'lost+found']
            data_url_results[data_url[0]] = {item for item in os.listdir(saved_dir) if item not in exclude_list}
        else:
            raise FileNotFoundError(f"The directory '{saved_dir}' does not exist.")
    return data_url_results


def relevant_segments_from_url_directories(url_directories):
    segments = {url.split('/')[-2] for url in url_directories}
    # Ignore if superseded
    return {url_segment for url_segment in segments if not is_superseded(url_segment, segments)}


def is_superseded(segment_id, all_segment_ids_set):
    try:
        return "_superseded" in segment_id or (str(int(segment_id) + 1) in all_segment_ids_set)
    except ValueError:
        return False


def get_all_scroll_segments(scroll_url):
    response = requests.get(scroll_url, auth=(username, password))

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")

        directories = []

        for link in soup.find_all("a"):
            href = link.get("href")

            if href:
                full_url = urljoin(scroll_url, href)
                parsed_url = urlparse(full_url)

                # Check if it's a directory (based on the path component of the URL)
                if parsed_url.path.endswith('/') and full_url.startswith(scroll_url):
                    directories.append(full_url)
        return directories
    else:
        print(f"Failed to fetch content from {scroll_url}. Status code: {response.status_code}")


def download_segment(base_dir, segment_to_get):
    def download_segment_helper(url_to_download):
        print(f"Downloading {segment_to_get}, from url {url_to_download}, saving to {base_dir}")
        # if os.path.exists(f"{base_dir}/{segment_to_get}"):
        #     print(f"Skipping {segment_to_get} because it already exists")
        #     return
        url_parts = url_to_download.rstrip('/').split('/')
        cut_dirs = len(url_parts) - 3  # Adjust the number of directories to cut as needed
        url_to_download = f"{urljoin(url_to_download, segment_to_get)}/"

        wget_command = [
            "wget",
            f"--user={username}",
            f"--password={password}",
            "-r",
            "-N",
            "--no-parent",
            "-A", "*.tif,mask.png",
            "-R", "*cellmap*, index.html*",
            "-nH",
            f"--cut-dirs={cut_dirs}",
            "-P", base_dir,
            f"{url_to_download}"
        ]
        result = subprocess.run(wget_command, check=True)
        if result.returncode == 0:
            directory_to_write = os.path.join(base_dir, segment_to_get)
            print(f"Successfully downloaded {segment_to_get} from {url_to_download}")
            with open(f"{directory_to_write}/base_url.txt", "w") as f:
                f.write(url_to_download)
        else:
            print(f"Failed to download {segment_to_get} from {url_to_download}")

    return download_segment_helper


if __name__ == "__main__":
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    for url_tuple in urls:
        print(f"Checking if directory {get_url_data_path(url_tuple)} exists")
        if not os.path.exists(get_url_data_path(url_tuple)):
            print(f"Creating directory {get_url_data_path(url_tuple)}")
            os.mkdir(get_url_data_path(url_tuple))
    # downloaded_segments = saved_segments(data_dir, urls)
    for url_tuple in urls:
        key, url = url_tuple
        scroll_segments = get_all_scroll_segments(url)
        segments_to_download = relevant_segments_from_url_directories(scroll_segments)
        print(f"For key {key}, segments to download: {segments_to_download}")
        with ThreadPoolExecutor(max_workers=max_concurrent_downloads) as executor:
            for segment in segments_to_download:
                executor.submit(download_segment(get_url_data_path(url_tuple), segment), url)
    # for segment in ["20230522181603", "20230702185752"]:
    #     download_segment("./data/scroll1_hari", segment)("http://dl.ash2txt.org/hari-seldon-uploads/team-finished-paths/scroll1/")
