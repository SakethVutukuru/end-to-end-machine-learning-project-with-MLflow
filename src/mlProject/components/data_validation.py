{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "from mlProject import logger\n",
    "from mlpro\n",
    "\n",
    "class DataValidation:\n",
    "    def __init__(self, config):\n",
    "        self.config = config\n",
    "\n",
    "    def validate_all_columns(self) -> bool:\n",
    "        try:\n",
    "            validation_status = True\n",
    "\n",
    "            data = pd.read_csv(self.config.unzip_data_dir)\n",
    "            all_cols = list(data.columns)\n",
    "\n",
    "            all_schema = self.config.all_schema\n",
    "\n",
    "            for col in all_cols:\n",
    "                if col not in all_schema:\n",
    "                    validation_status = False\n",
    "                    break  # Exit the loop as soon as a column is not in the schema\n",
    "                expected_data_type = all_schema[col]\n",
    "                actual_data_type = data[col].dtype\n",
    "                if expected_data_type != str(actual_data_type):\n",
    "                    validation_status = False\n",
    "                    break  # Exit the loop as soon as a data type mismatch is found\n",
    "\n",
    "            with open(self.config.STATUS_FILE, 'w') as f:\n",
    "                f.write(f\"Validation status: {validation_status}\")\n",
    "\n",
    "            return validation_status\n",
    "\n",
    "        except Exception as e:\n",
    "            raise e\n"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
