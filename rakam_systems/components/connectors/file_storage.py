import os
from pathlib import Path
import boto3

from rakam_systems.components.component import Component
from rakam_systems.system_manager import SystemManager

class S3FileManager(Component):
    """
    This class manages files in an S3 bucket.

    Attributes:
        s3_client (boto3.client): The S3 client object.
        bucket_name (str): The name of the S3 bucket.
        file_names (list): A list of file names in the bucket.
        folders (list): A list of folder names in the bucket.
    """

    def __init__(self, bucket_name, system_manager: SystemManager):
        """
        Initialize the S3FileManager object.

        Args:
            bucket_name (str): The name of the S3 bucket.
        """
        self.s3_client = None  # Delay S3 client initialization
        self.bucket_name = bucket_name
        self.file_names = []
        self.folders = []
        self.system_manager = system_manager

    def _initialize_s3_client(self):
        """Lazy initialization of the S3 client."""
        if self.s3_client is None:
            AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
            AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
            AWS_S3_REGION_NAME = os.getenv("AWS_S3_REGION_NAME")

            session = boto3.Session(
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_S3_REGION_NAME,
            )
            self.s3_client = session.client("s3")

    def get_file_folders(self, prefix=""):
        """
        Get the file and folder names in the S3 bucket.

        Args:
            prefix (str, optional): The prefix to filter the file and folder names. Defaults to "".

        Returns:
            tuple: A tuple of two lists, the first containing file names and the second containing folder names.
        """
        self.file_names = []  # Reset file names list
        self.folders = []      # Reset folders list
        self._initialize_s3_client()  # Ensure S3 client is initialized before use

        default_kwargs = {"Bucket": self.bucket_name, "Prefix": prefix}
        next_token = ""

        while next_token is not None:
            updated_kwargs = default_kwargs.copy()
            if next_token != "":
                updated_kwargs["ContinuationToken"] = next_token

            response = self.s3_client.list_objects_v2(**updated_kwargs)
            contents = response.get("Contents", [])

            for result in contents:
                key = result.get("Key")
                if key[-1] == "/":
                    self.folders.append(key)
                else:
                    self.file_names.append(key)

            next_token = response.get("NextContinuationToken")

        return self.file_names, self.folders

    def download_files(self, local_path=None):
        """
        Download files from the S3 bucket to the local machine.

        Args:
            local_path (str, optional): The local path to download the files. Defaults to the current working directory.
        """
        self._initialize_s3_client()  # Ensure S3 client is initialized before use

        if local_path is None:
            local_path = Path.cwd()
        local_path = Path(local_path)

        for folder in self.folders:
            folder_path = local_path / folder
            folder_path.mkdir(parents=True, exist_ok=True)

        for file_name in self.file_names:
            file_path = local_path / file_name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(self.bucket_name, file_name, str(file_path))

    def upload_folders(self, local_path, prefix="test"):
        """
        Upload folders from the local machine to the S3 bucket.

        Args:
            local_path (str): The local path of the folders to upload.
            prefix (str, optional): The prefix to add to the S3 file paths. Defaults to "test".
        """
        self._initialize_s3_client()  # Ensure S3 client is initialized before use

        for root, dirs, files in os.walk(local_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                s3_file_path = os.path.relpath(local_file_path, local_path).replace("\\", "/")
                s3_file_path = os.path.join(prefix, s3_file_path)
                self.s3_client.upload_file(local_file_path, self.bucket_name, s3_file_path)
            for dir in dirs:
                local_dir_path = os.path.join(root, dir)
                s3_dir_path = os.path.relpath(local_dir_path, local_path).replace("\\", "/")
                s3_dir_path = os.path.join(prefix, s3_dir_path) + "/"
                self.s3_client.put_object(Bucket=self.bucket_name, Key=s3_dir_path)

    def update_prefix(self, local_path, prefix):
        """
        Update the prefix folder in the S3 bucket by overwriting files with those from the local path.

        Args:
            local_path (str): The local path containing files to upload.
            prefix (str): The prefix in the S3 bucket where files will be updated.
        """
        self._initialize_s3_client()  # Ensure S3 client is initialized before use

        # First, list all files in the specified S3 prefix
        existing_files, _ = self.get_file_folders(prefix=prefix)
        existing_files_set = set(existing_files)

        # Collect local files to upload and map their S3 keys
        files_to_upload = []
        for root, _, files in os.walk(local_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                s3_file_key = os.path.join(prefix, os.path.relpath(local_file_path, local_path)).replace("\\", "/")
                files_to_upload.append((local_file_path, s3_file_key))

        # Determine files to delete in S3 (those that exist in S3 but not locally)
        local_files_set = set(file_key for _, file_key in files_to_upload)
        files_to_delete = existing_files_set - local_files_set

        # Delete files in S3 that are not in the local directory
        if files_to_delete:
            delete_keys = [{"Key": key} for key in files_to_delete]
            self.s3_client.delete_objects(Bucket=self.bucket_name, Delete={"Objects": delete_keys})

        # Upload local files, overwriting any existing files in the bucket
        for local_file_path, s3_file_key in files_to_upload:
            self.s3_client.upload_file(local_file_path, self.bucket_name, s3_file_key)

    def empty(self, prefix=""):
        """
        Delete all objects in the S3 bucket that match the given prefix.

        Args:
            prefix (str, optional): The prefix to filter the objects to delete. Defaults to "".
        """
        self._initialize_s3_client()  # Ensure S3 client is initialized before use

        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            objects = page.get("Contents", [])
            delete_keys = [{"Key": obj["Key"]} for obj in objects]
            
            if delete_keys:
                self.s3_client.delete_objects(Bucket=self.bucket_name, Delete={"Objects": delete_keys})

    def call_main(self, **kwargs) -> dict:
        pass

    def test(self, **kwargs) -> bool:
        pass
