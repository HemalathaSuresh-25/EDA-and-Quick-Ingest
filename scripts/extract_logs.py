import os
import tarfile
from datetime import datetime

# Input tar file and target extraction directory
tar_file = "C:/Users/hemalatha/Desktop/attest-eda/raw_logs/Attest_Archive_2025_Sep_22_10_25_01.tar.gz"
extract_path = "C:/Users/hemalatha/Desktop/attest-eda/data/standardized1"

# Open and process the archive
with tarfile.open(tar_file, "r:gz") as tar:
    print(f"Extracting logs from archive: {tar_file}")
    
    for member in tar.getmembers():
        # Extract only files (skip directories)
        if member.isfile():
            filename = os.path.basename(member.name)
            base, _ = os.path.splitext(filename)
            date_str = None

            # Detect date token (e.g., 20250922)
            for token in base.split("_"):
                if token.isdigit() and len(token) == 8:
                    date_str = token
                    break

            # Convert date string to YYYY-MM-DD format
            if date_str:
                try:
                    run_date = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
                except Exception:
                    run_date = "unknown_date"
            else:
                run_date = "unknown_date"

            # Detect suite name (3rd part of filename if available)
            parts = base.split("_")
            suite_name = parts[2] if len(parts) > 2 else "unknown_suite"

            # Create output folder structure
            target_dir = os.path.join(extract_path, run_date, suite_name)
            os.makedirs(target_dir, exist_ok=True)

            # Extract the log file into its folder
            tar.extract(member, target_dir)
            print(f"✔ Extracted: {filename} → {target_dir}")

print("\nAll logs successfully extracted to:", extract_path)
