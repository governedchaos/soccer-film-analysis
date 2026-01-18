"""
Soccer Film Analysis - Cloud Backup & Sync
Sync analysis data and projects across devices using cloud storage
"""

import os
import json
import hashlib
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
from loguru import logger


class SyncStatus(Enum):
    SYNCED = "synced"
    PENDING_UPLOAD = "pending_upload"
    PENDING_DOWNLOAD = "pending_download"
    CONFLICT = "conflict"
    ERROR = "error"


@dataclass
class SyncItem:
    """Represents a file to be synced"""
    local_path: str
    remote_path: str
    local_hash: Optional[str] = None
    remote_hash: Optional[str] = None
    local_modified: Optional[str] = None
    remote_modified: Optional[str] = None
    status: SyncStatus = SyncStatus.PENDING_UPLOAD
    size_bytes: int = 0


@dataclass
class SyncConfig:
    """Configuration for cloud sync"""
    provider: str = "local"  # local, google_drive, dropbox, onedrive
    remote_folder: str = "SoccerFilmAnalysis"
    auto_sync: bool = False
    sync_interval_minutes: int = 30
    sync_videos: bool = False  # Videos can be large
    sync_analysis: bool = True
    sync_database: bool = True
    sync_reports: bool = True
    last_sync: Optional[str] = None
    credentials_path: Optional[str] = None


class CloudProvider(ABC):
    """Abstract base class for cloud storage providers"""

    @abstractmethod
    def authenticate(self, credentials: Dict) -> bool:
        """Authenticate with the cloud service"""
        pass

    @abstractmethod
    def upload_file(self, local_path: Path, remote_path: str) -> bool:
        """Upload a file to cloud storage"""
        pass

    @abstractmethod
    def download_file(self, remote_path: str, local_path: Path) -> bool:
        """Download a file from cloud storage"""
        pass

    @abstractmethod
    def list_files(self, remote_folder: str) -> List[Dict]:
        """List files in a remote folder"""
        pass

    @abstractmethod
    def get_file_info(self, remote_path: str) -> Optional[Dict]:
        """Get metadata for a remote file"""
        pass

    @abstractmethod
    def delete_file(self, remote_path: str) -> bool:
        """Delete a file from cloud storage"""
        pass


class LocalFolderProvider(CloudProvider):
    """
    Local folder sync (for testing or network drives).
    Simulates cloud storage using a local folder.
    """

    def __init__(self, sync_folder: Optional[Path] = None):
        self.sync_folder = sync_folder or Path.home() / "SoccerFilmAnalysis_Sync"
        self.sync_folder.mkdir(parents=True, exist_ok=True)

    def authenticate(self, credentials: Dict) -> bool:
        return True  # No auth needed for local

    def upload_file(self, local_path: Path, remote_path: str) -> bool:
        try:
            dest = self.sync_folder / remote_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, dest)
            logger.debug(f"Uploaded {local_path} to {dest}")
            return True
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False

    def download_file(self, remote_path: str, local_path: Path) -> bool:
        try:
            source = self.sync_folder / remote_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, local_path)
            logger.debug(f"Downloaded {source} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False

    def list_files(self, remote_folder: str) -> List[Dict]:
        folder = self.sync_folder / remote_folder
        if not folder.exists():
            return []

        files = []
        for path in folder.rglob("*"):
            if path.is_file():
                rel_path = path.relative_to(self.sync_folder)
                stat = path.stat()
                files.append({
                    'path': str(rel_path),
                    'name': path.name,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'hash': self._calculate_hash(path)
                })
        return files

    def get_file_info(self, remote_path: str) -> Optional[Dict]:
        path = self.sync_folder / remote_path
        if not path.exists():
            return None
        stat = path.stat()
        return {
            'path': remote_path,
            'name': path.name,
            'size': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'hash': self._calculate_hash(path)
        }

    def delete_file(self, remote_path: str) -> bool:
        try:
            path = self.sync_folder / remote_path
            if path.exists():
                path.unlink()
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    def _calculate_hash(self, path: Path) -> str:
        """Calculate MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


class GoogleDriveProvider(CloudProvider):
    """
    Google Drive cloud storage provider.
    Requires google-api-python-client and oauth2client.
    """

    def __init__(self):
        self.service = None
        self.folder_id = None

    def authenticate(self, credentials: Dict) -> bool:
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build

            creds = None
            token_path = Path(credentials.get('token_path', 'token.json'))
            creds_path = Path(credentials.get('credentials_path', 'credentials.json'))

            if token_path.exists():
                creds = Credentials.from_authorized_user_file(str(token_path))

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    from google.auth.transport.requests import Request
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(creds_path),
                        ['https://www.googleapis.com/auth/drive.file']
                    )
                    creds = flow.run_local_server(port=0)
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())

            self.service = build('drive', 'v3', credentials=creds)
            logger.info("Google Drive authenticated")
            return True

        except ImportError:
            logger.error("Google Drive API libraries not installed")
            return False
        except Exception as e:
            logger.error(f"Google Drive auth failed: {e}")
            return False

    def _get_or_create_folder(self, folder_name: str) -> Optional[str]:
        """Get or create the sync folder in Google Drive"""
        if self.folder_id:
            return self.folder_id

        # Search for existing folder
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = self.service.files().list(q=query, fields="files(id)").execute()
        files = results.get('files', [])

        if files:
            self.folder_id = files[0]['id']
        else:
            # Create folder
            metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = self.service.files().create(body=metadata, fields='id').execute()
            self.folder_id = folder.get('id')

        return self.folder_id

    def upload_file(self, local_path: Path, remote_path: str) -> bool:
        try:
            from googleapiclient.http import MediaFileUpload

            folder_id = self._get_or_create_folder("SoccerFilmAnalysis")

            media = MediaFileUpload(str(local_path), resumable=True)
            metadata = {
                'name': Path(remote_path).name,
                'parents': [folder_id]
            }
            self.service.files().create(
                body=metadata,
                media_body=media,
                fields='id'
            ).execute()
            return True
        except Exception as e:
            logger.error(f"Google Drive upload failed: {e}")
            return False

    def download_file(self, remote_path: str, local_path: Path) -> bool:
        try:
            from googleapiclient.http import MediaIoBaseDownload
            import io

            # Find file by name
            query = f"name='{Path(remote_path).name}' and trashed=false"
            results = self.service.files().list(q=query, fields="files(id)").execute()
            files = results.get('files', [])

            if not files:
                return False

            file_id = files[0]['id']
            request = self.service.files().get_media(fileId=file_id)

            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()

            return True
        except Exception as e:
            logger.error(f"Google Drive download failed: {e}")
            return False

    def list_files(self, remote_folder: str) -> List[Dict]:
        try:
            folder_id = self._get_or_create_folder(remote_folder)
            query = f"'{folder_id}' in parents and trashed=false"
            results = self.service.files().list(
                q=query,
                fields="files(id, name, size, modifiedTime, md5Checksum)"
            ).execute()

            return [
                {
                    'path': f['name'],
                    'name': f['name'],
                    'size': int(f.get('size', 0)),
                    'modified': f.get('modifiedTime'),
                    'hash': f.get('md5Checksum')
                }
                for f in results.get('files', [])
            ]
        except Exception as e:
            logger.error(f"Google Drive list failed: {e}")
            return []

    def get_file_info(self, remote_path: str) -> Optional[Dict]:
        files = self.list_files("SoccerFilmAnalysis")
        for f in files:
            if f['name'] == Path(remote_path).name:
                return f
        return None

    def delete_file(self, remote_path: str) -> bool:
        try:
            query = f"name='{Path(remote_path).name}' and trashed=false"
            results = self.service.files().list(q=query, fields="files(id)").execute()
            files = results.get('files', [])
            if files:
                self.service.files().delete(fileId=files[0]['id']).execute()
            return True
        except Exception as e:
            logger.error(f"Google Drive delete failed: {e}")
            return False


class CloudSyncManager:
    """
    Manages synchronization between local data and cloud storage.
    """

    def __init__(self, config: Optional[SyncConfig] = None):
        from config import settings
        self.data_dir = settings.get_output_dir()
        self.config = config or SyncConfig()
        self.provider: Optional[CloudProvider] = None
        self.sync_state: Dict[str, SyncItem] = {}
        self._state_file = self.data_dir / ".sync_state.json"
        self._load_state()

    def set_provider(self, provider_name: str, credentials: Optional[Dict] = None) -> bool:
        """Set and authenticate with a cloud provider"""
        if provider_name == "local":
            self.provider = LocalFolderProvider()
        elif provider_name == "google_drive":
            self.provider = GoogleDriveProvider()
        else:
            logger.error(f"Unknown provider: {provider_name}")
            return False

        self.config.provider = provider_name

        if credentials:
            return self.provider.authenticate(credentials)
        return True

    def _load_state(self):
        """Load sync state from disk"""
        if self._state_file.exists():
            try:
                with open(self._state_file, 'r') as f:
                    data = json.load(f)
                    self.sync_state = {
                        k: SyncItem(**v) for k, v in data.get('items', {}).items()
                    }
                    if 'config' in data:
                        self.config = SyncConfig(**data['config'])
            except Exception as e:
                logger.warning(f"Failed to load sync state: {e}")

    def _save_state(self):
        """Save sync state to disk"""
        try:
            data = {
                'items': {k: asdict(v) for k, v in self.sync_state.items()},
                'config': asdict(self.config)
            }
            with open(self._state_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save sync state: {e}")

    def _calculate_hash(self, path: Path) -> str:
        """Calculate MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def scan_local_files(self) -> List[SyncItem]:
        """Scan local files that should be synced"""
        items = []

        patterns = []
        if self.config.sync_analysis:
            patterns.extend(["*.json", "*.pkl", "analysis_*.json"])
        if self.config.sync_database:
            patterns.extend(["*.db", "team_database.db"])
        if self.config.sync_reports:
            patterns.extend(["*.pdf", "*.txt", "*_report.*"])
        if self.config.sync_videos:
            patterns.extend(["*.mp4", "*.avi", "*.mov"])

        for pattern in patterns:
            for path in self.data_dir.glob(pattern):
                if path.is_file() and not path.name.startswith('.'):
                    rel_path = path.relative_to(self.data_dir)
                    local_hash = self._calculate_hash(path)
                    stat = path.stat()

                    item = SyncItem(
                        local_path=str(path),
                        remote_path=str(rel_path),
                        local_hash=local_hash,
                        local_modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        size_bytes=stat.st_size
                    )
                    items.append(item)

        return items

    def compare_with_remote(self) -> Dict[str, List[SyncItem]]:
        """Compare local files with remote and categorize by action needed"""
        if not self.provider:
            logger.error("No cloud provider configured")
            return {}

        local_files = {item.remote_path: item for item in self.scan_local_files()}
        remote_files = self.provider.list_files(self.config.remote_folder)
        remote_map = {f['path']: f for f in remote_files}

        to_upload = []
        to_download = []
        conflicts = []
        synced = []

        # Check local files
        for path, local_item in local_files.items():
            if path in remote_map:
                remote = remote_map[path]
                local_item.remote_hash = remote.get('hash')
                local_item.remote_modified = remote.get('modified')

                if local_item.local_hash == local_item.remote_hash:
                    local_item.status = SyncStatus.SYNCED
                    synced.append(local_item)
                elif local_item.local_modified > local_item.remote_modified:
                    local_item.status = SyncStatus.PENDING_UPLOAD
                    to_upload.append(local_item)
                else:
                    local_item.status = SyncStatus.PENDING_DOWNLOAD
                    to_download.append(local_item)
            else:
                local_item.status = SyncStatus.PENDING_UPLOAD
                to_upload.append(local_item)

        # Check remote-only files
        for path, remote in remote_map.items():
            if path not in local_files:
                item = SyncItem(
                    local_path=str(self.data_dir / path),
                    remote_path=path,
                    remote_hash=remote.get('hash'),
                    remote_modified=remote.get('modified'),
                    size_bytes=remote.get('size', 0),
                    status=SyncStatus.PENDING_DOWNLOAD
                )
                to_download.append(item)

        return {
            'upload': to_upload,
            'download': to_download,
            'conflicts': conflicts,
            'synced': synced
        }

    def sync(
        self,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict:
        """
        Perform full sync with cloud storage.

        Args:
            progress_callback: Called with (message, percentage)

        Returns:
            Dict with sync results
        """
        if not self.provider:
            return {'error': 'No cloud provider configured'}

        comparison = self.compare_with_remote()

        results = {
            'uploaded': 0,
            'downloaded': 0,
            'conflicts': len(comparison.get('conflicts', [])),
            'errors': []
        }

        total_items = len(comparison.get('upload', [])) + len(comparison.get('download', []))
        processed = 0

        # Upload local changes
        for item in comparison.get('upload', []):
            if progress_callback:
                progress_callback(f"Uploading {item.remote_path}", (processed / max(1, total_items)) * 100)

            if self.provider.upload_file(Path(item.local_path), item.remote_path):
                item.status = SyncStatus.SYNCED
                results['uploaded'] += 1
            else:
                item.status = SyncStatus.ERROR
                results['errors'].append(f"Failed to upload: {item.remote_path}")

            self.sync_state[item.remote_path] = item
            processed += 1

        # Download remote changes
        for item in comparison.get('download', []):
            if progress_callback:
                progress_callback(f"Downloading {item.remote_path}", (processed / max(1, total_items)) * 100)

            if self.provider.download_file(item.remote_path, Path(item.local_path)):
                item.status = SyncStatus.SYNCED
                results['downloaded'] += 1
            else:
                item.status = SyncStatus.ERROR
                results['errors'].append(f"Failed to download: {item.remote_path}")

            self.sync_state[item.remote_path] = item
            processed += 1

        self.config.last_sync = datetime.now().isoformat()
        self._save_state()

        if progress_callback:
            progress_callback("Sync complete", 100)

        logger.info(f"Sync complete: {results['uploaded']} uploaded, {results['downloaded']} downloaded")
        return results

    def create_backup(self, backup_name: Optional[str] = None) -> Optional[Path]:
        """Create a compressed backup of all data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = backup_name or f"backup_{timestamp}"
        backup_path = self.data_dir / f"{backup_name}.zip"

        try:
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for item in self.scan_local_files():
                    zf.write(item.local_path, item.remote_path)

            logger.info(f"Backup created: {backup_path}")

            # Upload backup to cloud if provider available
            if self.provider:
                self.provider.upload_file(backup_path, f"backups/{backup_path.name}")

            return backup_path
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return None

    def restore_backup(self, backup_path: Path) -> bool:
        """Restore data from a backup file"""
        try:
            with zipfile.ZipFile(backup_path, 'r') as zf:
                zf.extractall(self.data_dir)
            logger.info(f"Restored from backup: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False

    def get_sync_status(self) -> Dict:
        """Get current sync status summary"""
        comparison = self.compare_with_remote() if self.provider else {}
        return {
            'provider': self.config.provider,
            'last_sync': self.config.last_sync,
            'pending_upload': len(comparison.get('upload', [])),
            'pending_download': len(comparison.get('download', [])),
            'synced': len(comparison.get('synced', [])),
            'conflicts': len(comparison.get('conflicts', []))
        }
