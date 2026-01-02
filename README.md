# APT Repository Generator

[![Update APT Repository](https://github.com/bordeux/apt-repo/actions/workflows/update-repo.yml/badge.svg)](https://github.com/bordeux/apt-repo/actions/workflows/update-repo.yml)

Generate an APT repository from GitHub releases containing `.deb` packages. Host your own Debian/Ubuntu package repository on GitHub Pages.

## How It Works

1. Define projects in `projects.yaml` pointing to GitHub repositories that release `.deb` files
2. GitHub Actions fetches releases, downloads `.deb` packages, and generates APT repository metadata
3. The repository is deployed to GitHub Pages
4. Users can add your repository to their system and install packages with `apt`

## Quick Start

### 1. Configure Projects

Edit `projects.yaml` to add GitHub repositories:

```yaml
settings:
  codename: stable
  architectures:
    - amd64
    - arm64

projects:
  - repo: bordeux/tmpltool
    keep_versions: 1        # Keep 1 previous version
```

### 2. Enable GitHub Pages

1. Go to repository **Settings** > **Pages**
2. Set **Source** to "GitHub Actions"

### 3. (Optional) Set Up GPG Signing

For signed repositories:

1. Generate a GPG key: `gpg --full-generate-key`
2. Export private key: `gpg --armor --export-secret-keys YOUR_KEY_ID`
3. Add as repository secret `GPG_PRIVATE_KEY`
4. Add repository variable `GPG_KEY_ID` with your key ID

### 4. Trigger the Workflow

- Push changes to `projects.yaml`, or
- Go to **Actions** > **Update APT Repository** > **Run workflow**

## Using the Repository

Once deployed, users can add your repository:

```bash
# Download and install the GPG key (if signed)
curl -fsSL https://bordeux.github.io/apt-repo/public.key | sudo gpg --dearmor -o /etc/apt/keyrings/bordeux.gpg

# Add the repository
echo "deb [signed-by=/etc/apt/keyrings/bordeux.gpg] https://bordeux.github.io/apt-repo stable main" | sudo tee /etc/apt/sources.list.d/bordeux.list

# Update and install
sudo apt update
sudo apt install tmpltool
```

For unsigned repositories, omit the `signed-by` option:

```bash
echo "deb [trusted=yes] https://bordeux.github.io/apt-repo stable main" | sudo tee /etc/apt/sources.list.d/bordeux.list
```

## Configuration Reference

### projects.yaml

```yaml
settings:
  codename: stable              # Distribution codename (default: stable)
  components:                   # Repository components
    - main
  architectures:                # Supported architectures
    - amd64
    - arm64
  label: "Bordeux Packages"     # Repository label
  origin: "bordeux"             # Repository origin

projects:
  - repo: bordeux/tmpltool      # GitHub repository (required)
    name: tmpltool              # Package name override (default: repo name)
    description: "Description"  # Description override
    keep_versions: 1            # Past versions to keep (default: 0)
    asset_pattern: ".*amd64.*"  # Regex to filter .deb assets
```

### Command Line

```bash
# Set up virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Generate repository locally
python scripts/generate_repo.py --output repo

# Process specific project
python scripts/generate_repo.py --project bordeux/tmpltool

# Dry run (no downloads)
python scripts/generate_repo.py --dry-run

# List configured projects
python scripts/generate_repo.py --list

# With GPG signing
python scripts/generate_repo.py --gpg-key YOUR_KEY_ID

# Skip signing
python scripts/generate_repo.py --no-sign
```

## Repository Structure

Generated repository layout:

```
repo/
├── dists/
│   └── stable/
│       ├── main/
│       │   ├── binary-amd64/
│       │   │   ├── Packages
│       │   │   └── Packages.gz
│       │   └── binary-arm64/
│       │       ├── Packages
│       │       └── Packages.gz
│       ├── Release
│       ├── Release.gpg      (if signed)
│       └── InRelease        (if signed)
├── pool/
│   └── main/
│       └── *.deb            (downloaded packages)
└── public.key               (if signed)
```

## Requirements

- Python 3.9+
- PyYAML
- `dpkg-deb` or `ar` + `tar` (for extracting package metadata)
- GPG (optional, for signing)

## License

MIT