"""
ASL Data Collection Web App
============================
A Flask web app that lets multiple users contribute hand sign images
for fine-tuning your ASL model.

Features:
- Users enter their name, resume where they left off
- Shows reference image of the sign to make
- Captures hand crops via browser webcam + MediaPipe JS
- Uploads images to Google Drive via API (organized by user/letter)
- Tracks progress per user in a local JSON file

Setup:
    1. pip install flask google-api-python-client google-auth-oauthlib
    2. Place your Google service account JSON key as 'service_account.json'
       (or set GOOGLE_SERVICE_ACCOUNT_PATH env var)
    3. Create a 'reference_images' folder with A.jpg–Z.jpg showing each sign
    4. Set DRIVE_FOLDER_ID to your "finetune data" folder's ID in Google Drive
    5. python app.py

Google Drive Setup:
    1. Go to console.cloud.google.com → create project
    2. Enable "Google Drive API"
    3. Create a Service Account → download JSON key
    4. Share your "finetune data" Drive folder with the service account email
       (the email looks like: name@project.iam.gserviceaccount.com)
"""

import os
import json
import time
import base64
import threading
import queue

from flask import Flask, render_template, request, jsonify, send_from_directory

# Google Drive imports
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaInMemoryUpload

# =============================================================================
# CONFIG
# =============================================================================
DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID", "19c9nHCyjB_bq3lVKNu7QsIbsx8zKBKyR")

# Path to OAuth2 client secret JSON (download from Google Cloud Console)
# Go to: APIs & Services → Credentials → Create OAuth 2.0 Client ID → Desktop App
CLIENT_SECRET_PATH = os.environ.get("CLIENT_SECRET_PATH", "client_secret.json")

# Token file (auto-created after first login)
TOKEN_PATH = "token.json"

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]

# Folder containing reference images (A.jpg–Z.jpg) showing what sign to make
REFERENCE_IMAGES_DIR = "input_data"

# How many images to collect per letter
TARGET_PER_SYMBOL = 250

# All symbols to collect
SYMBOLS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z","SPACE","DEL"
    ]

# Local file to track user progress
PROGRESS_FILE = "user_progress.json"

# =============================================================================
# APP SETUP
# =============================================================================
app = Flask(__name__)

# Thread lock for progress file writes
progress_lock = threading.Lock()

# Upload queue — Drive uploads happen sequentially in a background thread
upload_queue = queue.Queue()

# Cache for Drive folder IDs so we don't look them up every single upload
_folder_id_cache = {}  # key: "parent_id/name" → value: folder_id


# =============================================================================
# GOOGLE DRIVE HELPERS
# =============================================================================
_drive_service = None


def get_drive_service():
    """Lazy-init Google Drive API using OAuth2 (your personal account)."""
    global _drive_service
    if _drive_service is not None:
        return _drive_service

    if not os.path.exists(CLIENT_SECRET_PATH):
        print(f"WARNING: OAuth client secret not found at {CLIENT_SECRET_PATH}")
        print("  Download it from Google Cloud Console → APIs & Services → Credentials")
        print("  Create an OAuth 2.0 Client ID (Desktop App), download the JSON.")
        print("  Images will only be saved locally.")
        return None

    creds = None
    # Load saved token if it exists
    if os.path.exists(TOKEN_PATH):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, DRIVE_SCOPES)
        except Exception:
            creds = None

    # If no valid creds, do the OAuth login flow (opens browser once)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None

        if not creds:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRET_PATH, DRIVE_SCOPES
            )
            creds = flow.run_local_server(port=8090)

        # Save token for next time
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())

    try:
        _drive_service = build("drive", "v3", credentials=creds)
        print("Google Drive API connected (OAuth2).")
        return _drive_service
    except Exception as e:
        print(f"Google Drive build failed: {e}")
        return None


def find_or_create_folder(service, name, parent_id):
    """Find a subfolder by name inside a parent, or create it. Uses cache."""
    cache_key = f"{parent_id}/{name}"
    if cache_key in _folder_id_cache:
        return _folder_id_cache[cache_key]

    query = (
        f"name = '{name}' and "
        f"'{parent_id}' in parents and "
        f"mimeType = 'application/vnd.google-apps.folder' and "
        f"trashed = false"
    )
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])

    if files:
        _folder_id_cache[cache_key] = files[0]["id"]
        return files[0]["id"]

    # Create it
    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    folder = service.files().create(body=metadata, fields="id").execute()
    _folder_id_cache[cache_key] = folder["id"]
    return folder["id"]


def upload_to_drive(service, folder_id, filename, image_bytes, mime="image/jpeg"):
    """Upload an image to a specific Drive folder."""
    metadata = {"name": filename, "parents": [folder_id]}
    media = MediaInMemoryUpload(image_bytes, mimetype=mime)
    service.files().create(body=metadata, media_body=media, fields="id").execute()


# =============================================================================
# PROGRESS TRACKING
# =============================================================================
def load_progress():
    """Load user progress from JSON file."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_progress(data):
    """Save user progress to JSON file."""
    with progress_lock:
        with open(PROGRESS_FILE, "w") as f:
            json.dump(data, f, indent=2)


def get_user_progress(user_key):
    """Get a specific user's progress. Returns dict of {letter: count}."""
    progress = load_progress()
    if user_key not in progress:
        progress[user_key] = {s: 0 for s in SYMBOLS}
        save_progress(progress)
    return progress[user_key]


def increment_user_count(user_key, letter):
    """Increment the count for a user's letter and return new count."""
    with progress_lock:
        progress = load_progress()
        if user_key not in progress:
            progress[user_key] = {s: 0 for s in SYMBOLS}
        progress[user_key][letter] = progress[user_key].get(letter, 0) + 1
        count = progress[user_key][letter]
        with open(PROGRESS_FILE, "w") as f:
            json.dump(progress, f, indent=2)
    return count


# =============================================================================
# ROUTES
# =============================================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/collect")
def collect():
    first = request.args.get("first", "").strip()
    last = request.args.get("last", "").strip()
    if not first or not last:
        return "Missing name", 400
    user_key = f"{first.lower()}_{last.lower()}"
    return render_template(
        "collect.html",
        first=first,
        last=last,
        user_key=user_key,
        symbols=SYMBOLS,
        target=TARGET_PER_SYMBOL,
    )


@app.route("/api/progress/<user_key>")
def api_progress(user_key):
    """Get user's progress for all letters."""
    progress = get_user_progress(user_key)
    # Find the first incomplete letter
    current_letter = None
    for s in SYMBOLS:
        if progress.get(s, 0) < TARGET_PER_SYMBOL:
            current_letter = s
            break
    return jsonify({
        "progress": progress,
        "current_letter": current_letter,
        "target": TARGET_PER_SYMBOL,
        "complete": current_letter is None,
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Receive a cropped hand image and save it."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400

    user_key = data.get("user_key", "")
    letter = data.get("letter", "")
    image_b64 = data.get("image", "")

    if not user_key or not letter or not image_b64:
        return jsonify({"error": "Missing fields"}), 400

    # Decode base64 image
    try:
        # Strip data URL prefix if present
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_b64)
    except Exception as e:
        return jsonify({"error": f"Bad image data: {e}"}), 400

    timestamp = int(time.time() * 1000)
    filename = f"{letter}_{timestamp}.jpg"

    # Queue for Google Drive upload (happens in background thread)
    if DRIVE_FOLDER_ID != "YOUR_FOLDER_ID_HERE":
        upload_queue.put({
            "user_key": user_key,
            "letter": letter,
            "filename": filename,
            "image_bytes": image_bytes,
        })

    # Update progress
    new_count = increment_user_count(user_key, letter)

    return jsonify({
        "success": True,
        "count": new_count,
        "target": TARGET_PER_SYMBOL,
    })


@app.route("/reference/<letter>")
def reference_image(letter):
    """Serve a reference image for a given letter."""
    letter = letter.upper()
    # Try multiple extensions
    for ext in ["jpg", "png", "jpeg"]:
        filepath = os.path.join(REFERENCE_IMAGES_DIR, f"{letter}.{ext}")
        if os.path.exists(filepath):
            return send_from_directory(
                REFERENCE_IMAGES_DIR, f"{letter}.{ext}"
            )
    # Return a 404 placeholder
    return "Reference image not found", 404


# =============================================================================
# BACKGROUND DRIVE UPLOAD WORKER
# =============================================================================
def drive_upload_worker():
    """Process upload queue one at a time to avoid SSL concurrency issues."""
    while True:
        item = upload_queue.get()
        if item is None:
            break

        service = get_drive_service()
        if not service:
            upload_queue.task_done()
            continue

        try:
            user_folder_id = find_or_create_folder(
                service, item["user_key"], DRIVE_FOLDER_ID
            )
            letter_folder_id = find_or_create_folder(
                service, item["letter"], user_folder_id
            )
            upload_to_drive(
                service, letter_folder_id,
                item["filename"], item["image_bytes"]
            )
        except Exception as e:
            print(f"Drive upload error ({item['filename']}): {e}")
            # Retry once after a short pause
            try:
                time.sleep(0.5)
                upload_to_drive(
                    service, letter_folder_id,
                    item["filename"], item["image_bytes"]
                )
            except Exception:
                print(f"  Retry failed — image saved locally only.")

        upload_queue.task_done()


# =============================================================================
# START UPLOAD WORKER (at module level so it works with Flask reloader)
# =============================================================================
os.makedirs("input_data", exist_ok=True)  # ensure reference images dir exists
_upload_worker = threading.Thread(target=drive_upload_worker, daemon=True)
_upload_worker.start()


# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    print(f"\nASL Data Collection Server")
    print(f"  Reference images: {REFERENCE_IMAGES_DIR}")
    print(f"  Drive folder ID:  {DRIVE_FOLDER_ID}")
    print(f"  Target per letter: {TARGET_PER_SYMBOL}")
    print(f"  Progress file:    {PROGRESS_FILE}")
    print(f"\n  Open http://localhost:5000 in your browser\n")
    app.run(host="0.0.0.0", port=5000, debug=True, ssl_context=('cert.pem', 'key.pem'))
