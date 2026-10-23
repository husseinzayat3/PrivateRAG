# Google Drive

PrivateRAG can index the files in a Google Drive folder alongside your local files. It asks
for **read-only** access (`drive.readonly`), and the documents are downloaded straight to your
machine: Google sees an app reading your Drive, not your questions or answers.

## What gets indexed

- **Google Docs, Sheets and Slides**, exported as text (Sheets as one document per row).
- **PDFs.**
- Only files directly in the folder you give; subfolders aren't included yet.

Other file types in the folder are skipped.

## One-time setup

You need your own **OAuth client ID** of type **Desktop app**. It's free and takes about five
minutes.

### 1. Enable the API

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a project (or select an existing one).
3. Go to **APIs & Services → Library**, search for **Google Drive API** and click **Enable**.

### 2. Configure the OAuth consent screen

1. Go to **APIs & Services → OAuth consent screen**.
2. Choose **External** and click **Create**.
3. Fill in the required fields (app name, support email, developer contact).
4. Under **Scopes**, click **Add or Remove Scopes** and select `.../auth/drive.readonly`.
5. Under **Test users**, add your own Google account.

### 3. Create the credentials

1. Go to **APIs & Services → Credentials**.
2. Click **+ Create Credentials → OAuth client ID**.
3. Choose **Desktop app** as the application type and give it a name.
4. Click **Download JSON**, rename the file to `credentials.json`, and put it in the project
   root (next to `app.py`), or set `GDRIVE_CREDENTIALS_PATH` to its location.

## Ingesting a folder

1. Restart the app; the sidebar shows *Google Drive credentials found*.
2. Open the folder in Google Drive and copy its ID from the URL:
   `https://drive.google.com/drive/folders/`**`1AbCdEfGhIjKlMnOpQrStUv`**
3. Paste the ID into **Google Drive Folder ID** and click **Ingest Documents**.

The first time, a browser window asks you to sign in and allow read-only access. Google will
probably warn that it **hasn't verified this app**. That's expected for your own OAuth client:
click **Advanced → Go to (app name)**. Afterwards a `token.json` is saved, so you won't be asked
again. If an existing `token.json` was issued for a different scope, you're asked to sign in
again.

Ingesting the same folder again only adds what changed, and edited files have their old
chunks replaced, just like local files.

## Using Google Drive with Docker

The sign-in opens a browser from the app, which doesn't work inside a container. See
[Troubleshooting → Using Google Drive with Docker](TROUBLESHOOTING.md#using-google-drive-with-docker)
for the workaround: sign in once outside Docker, then mount the token.

## Security notes

- `credentials.json` identifies your OAuth app, and `token.json` grants read access to your
  whole Drive. Keep both private. They're in `.gitignore` and `.dockerignore`.
- To revoke access, delete `token.json` and remove the app at
  [myaccount.google.com/permissions](https://myaccount.google.com/permissions).
