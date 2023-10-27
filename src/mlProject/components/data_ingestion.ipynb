{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import urllib.request as request\n",
    "import zipfile\n",
    "from mlProject import logger\n",
    "from mlProject.utils.common import get_size\n",
    "from pathlib import Path\n",
    "from mlProject.entity.config_entity import (DataIngestionConfig)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "class DataIngestion:\n",
    "    def __init__(self, config: DataIngestionConfig):\n",
    "        self.config = config\n",
    "\n",
    "\n",
    "    \n",
    "    def download_file(self):\n",
    "        if not os.path.exists(self.config.local_data_file):\n",
    "            filename, headers = request.urlretrieve(\n",
    "                url = self.config.source_URL,\n",
    "                filename = self.config.local_data_file\n",
    "            )\n",
    "            logger.info(f\"{filename} download! with following info: \\n{headers}\")\n",
    "        else:\n",
    "            logger.info(f\"File already exists of size: {get_size(Path(self.config.local_data_file))}\")\n",
    "\n",
    "\n",
    "\n",
    "    def extract_zip_file(self):\n",
    "        \"\"\"\n",
    "        zip_file_path: str\n",
    "        Extracts the zip file into the data directory\n",
    "        Function returns None\n",
    "        \"\"\"\n",
    "        unzip_path = self.config.unzip_dir\n",
    "        os.makedirs(unzip_path, exist_ok=True)\n",
    "        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:\n",
    "            zip_ref.extractall(unzip_path)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "mlproj",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
