"""
File: model_downloader.py
Description: Automated retrieval module for Large Language Models (LLMs) and Embedding models.
             Implements robust downloading mechanisms via ModelScope Hub with enhanced
             error handling, automatic retries, and network stability adjustments.
Environment: Python 3.12. Requires `modelscope` package.
"""

import os
import sys
import time
import logging
from typing import Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

class RobustOfflineDownloader:
    """
    Encapsulates the core downloading logic.
    Utilizes ModelScope Hub as an alternative artifact repository for network-constrained environments.
    """
    def __init__(self, target_dir: str = "local_models"):
        """
        Initializes the downloader with a defined local storage strategy.
        """
        self.target_dir = os.path.abspath(target_dir)
        self._setup_environment()

    def _setup_environment(self) -> None:
        """Provisions and validates the local storage directory structure."""
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)
            logging.info(f"Initialized storage directory: {self.target_dir}")

    def download_model(self, ms_repo_id: str, hf_repo_id: str, local_folder_name: str, max_retries: int = 5, retry_delay: int = 10) -> bool:
        """
        Executes the artifact retrieval process from ModelScope with built-in retry logic
        for handling hash validation failures and network timeouts.

        Args:
            ms_repo_id (str): Repository identifier on ModelScope.
            hf_repo_id (str): Original HuggingFace identifier (for logging reference).
            local_folder_name (str): Target sub-directory name.
            max_retries (int): Maximum number of retry attempts for failed downloads.
            retry_delay (int): Seconds to wait before retrying.
            
        Returns:
            bool: Process execution status (True if successful, False otherwise).
        """
        try:
            # Late import to prevent initial dependency faults if module is missing
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError:
            logging.error("[DEPENDENCY FAULT] Missing 'modelscope' module. Execute: pip install modelscope")
            sys.exit(1)

        destination_path = os.path.join(self.target_dir, local_folder_name)
        logging.info(f"Initiating retrieval for: {ms_repo_id} (HF Mapping: {hf_repo_id})")
        logging.info(f"Target destination: {destination_path}")

        for attempt in range(1, max_retries + 1):
            try:
                logging.info(f"Download attempt {attempt}/{max_retries} for {ms_repo_id}...")
                
                # Execute download. Reduced max_workers for better stability on unstable networks.
                snapshot_download(
                    model_id=ms_repo_id,
                    local_dir=destination_path,
                    max_workers=2, # Giảm xuống 2 để tránh đứt gãy đường truyền (có thể chỉnh thành 1 nếu mạng quá yếu)
                    revision='master'
                )
                
                logging.info(f"Artifact retrieval successfully completed for: {local_folder_name}\n")
                return True

            except Exception as e:
                error_msg = str(e)
                logging.warning(f"[NETWORK/SYSTEM ISSUE] Attempt {attempt} failed: {error_msg}")
                
                if attempt < max_retries:
                    logging.info(f"Waiting {retry_delay} seconds before retrying...")
                    time.sleep(retry_delay)
                else:
                    logging.error(f"[SYSTEM FAULT] Final failure during artifact retrieval for {ms_repo_id} after {max_retries} attempts.\n")
                    return False
        
        return False

def main() -> None:
    """
    Orchestrates the downloading process based on a predefined repository mapping matrix.
    """
    
    # Repository Mapping Matrix (HuggingFace ID -> (ModelScope ID, Local Folder Name))
    target_models: Dict[str, Tuple[str, str]] = {
        # Target Small Language Models (SLMs)
        #"sentence-transformers/all-MiniLM-L6-v2": ("AI-ModelScope/all-MiniLM-L6-v2", "all-MiniLM-L6-v2"),
        "microsoft/Phi-3.5-mini-instruct": ("LLM-Research/Phi-3.5-mini-instruct", "Phi-3.5-mini-instruct"),
        "Qwen/Qwen2.5-3B-Instruct": ("qwen/Qwen2.5-3B-Instruct", "Qwen2.5-3B-Instruct")
    }

    downloader = RobustOfflineDownloader(target_dir="local_models")
    
    success_count = 0
    total_models = len(target_models)

    logging.info("==================================================")
    logging.info("INITIALIZING MODELSCOPE ARTIFACT RETRIEVAL ENGINE")
    logging.info("==================================================")

    for hf_repo, (ms_repo, folder_name) in target_models.items():
        status = downloader.download_model(
            ms_repo_id=ms_repo, 
            hf_repo_id=hf_repo,
            local_folder_name=folder_name,
            max_retries=3,
            retry_delay=10
        )
        if status:
            success_count += 1

    logging.info("==================================================")
    logging.info(f"PIPELINE TERMINATED. Success Rate: {success_count}/{total_models}")
    logging.info("==================================================")

if __name__ == "__main__":
    main()