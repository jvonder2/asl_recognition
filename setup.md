# ASL Data Collection Web App — Setup Guide

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in your browser.
Share your IP address with others on the same network (e.g., `http://192.168.1.x:5000`).

---

## Google Drive API Setup (one-time)

### 1. Create a Google Cloud Project
1. Go to [console.cloud.google.com](https://console.cloud.google.com/)
2. Click **"New Project"** → name it something like "ASL Collector"
3. Select the project

### 2. Enable the Drive API
1. Go to **APIs & Services → Library**
2. Search for **"Google Drive API"**
3. Click **Enable**

### 3. Create a Service Account
1. Go to **APIs & Services → Credentials**
2. Click **"+ CREATE CREDENTIALS" → "Service account"**
3. Name it (e.g., "asl-uploader"), click **Create and Continue**
4. Skip the optional steps, click **Done**
5. Click on the service account you just created
6. Go to the **Keys** tab → **Add Key → Create new key → JSON**
7. Save the downloaded file as **`service_account.json`** in the same folder as `app.py`

### 4. Share Your Drive Folder
1. In Google Drive, create a folder called **"finetune data"**
2. Right-click → **Share**
3. Paste the service account email (looks like `name@project.iam.gserviceaccount.com`)
4. Give it **Editor** access
5. Copy the folder ID from the URL:
   `https://drive.google.com/drive/folders/` **`THIS_PART_IS_THE_ID`**

### 5. Configure the App
Edit `app.py` and set:
```python
DRIVE_FOLDER_ID = "paste_your_folder_id_here"
```

Or use environment variables:
```bash
set DRIVE_FOLDER_ID=paste_your_folder_id_here
set GOOGLE_SERVICE_ACCOUNT_PATH=service_account.json
python app.py
```

---

## Reference Images

Create a folder called `reference_images` (in the same directory as `app.py`)
and add images showing each ASL hand sign:

```
reference_images/
  A.jpg
  B.jpg
  C.jpg
  ...
  Z.jpg
```

Or set a custom path:
```python
REFERENCE_IMAGES_DIR = r"C:\path\to\your\reference\images"
```

---

## How It Works

### For Users
1. User visits the website, enters their name
2. If they've been here before, it resumes at the letter they left off
3. They see a reference image of the sign to make
4. Press **Start** → hold the sign → images auto-capture every 200ms
5. After 250 images, it auto-advances to the next letter
6. They can pause, stop, or skip any letter

### Data Organization

**Google Drive:**
```
finetune data/
  john_doe/
    A/
      A_1234567890.jpg
      A_1234567891.jpg
      ...
    B/
      B_1234567892.jpg
      ...
  jane_smith/
    A/
      ...
```

**Local Backup** (in case Drive upload fails):
```
local_backup/
  john_doe/
    A/
      A_1234567890.jpg
      ...
```

**Progress Tracking:**
- Stored in `user_progress.json` (auto-created)
- Tracks how many images each user has submitted per letter
- Users can close the browser and resume later

---

## Hosting for Others

### Same Network (easiest)
The app runs on `0.0.0.0:5000`, so anyone on your Wi-Fi can access it at
`http://YOUR_IP:5000`. Find your IP with `ipconfig` (Windows) or `ifconfig` (Mac/Linux).

### Internet Access (optional)
For access outside your network, use a tunnel service:
```bash
# Using ngrok (free)
ngrok http 5000
```
This gives you a public URL like `https://abc123.ngrok.io` that anyone can use.

---

## Config Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DRIVE_FOLDER_ID` | `YOUR_FOLDER_ID_HERE` | Google Drive folder ID |
| `SERVICE_ACCOUNT_PATH` | `service_account.json` | Path to service account key |
| `REFERENCE_IMAGES_DIR` | `C:\...\reference_images` | Folder with A.jpg–Z.jpg |
| `TARGET_PER_SYMBOL` | `250` | Images to collect per letter |
| `LOCAL_BACKUP_DIR` | `local_backup` | Local backup folder |
| `PROGRESS_FILE` | `user_progress.json` | User progress tracker |