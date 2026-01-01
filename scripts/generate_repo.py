#!/usr/bin/env python3
"""
APT Repository Generator

Generates an APT repository from GitHub releases containing .deb packages.
Similar to homebrew-tap but for Debian-based systems.

Supports incremental updates - can update a single project while preserving others.
"""

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import yaml


# Architecture patterns for .deb files
ARCH_PATTERNS = {
    "amd64": [r"amd64", r"x86_64", r"x64"],
    "arm64": [r"arm64", r"aarch64"],
    "i386": [r"i386", r"i686", r"x86[^_]"],
    "armhf": [r"armhf", r"armv7"],
}

# Manifest file to track packages by project
MANIFEST_FILE = "packages.json"


@dataclass
class DebPackage:
    """Represents a .deb package from a GitHub release."""
    name: str
    version: str
    architecture: str
    url: str
    filename: str
    project_repo: str = ""  # Track which project this belongs to
    size: int = 0
    sha256: str = ""
    sha512: str = ""
    md5: str = ""
    # Extracted from .deb control file
    description: str = ""
    maintainer: str = ""
    depends: str = ""
    section: str = "misc"
    priority: str = "optional"
    homepage: str = ""


@dataclass
class Release:
    """Represents a GitHub release."""
    tag: str
    version: str
    major_minor: str
    packages: list[DebPackage] = field(default_factory=list)


@dataclass
class Project:
    """Project configuration from projects.yaml."""
    repo: str
    name: str = ""
    description: str = ""
    keep_versions: int = 0
    asset_pattern: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = self.repo.split("/")[-1]


@dataclass
class RepoSettings:
    """Repository settings from projects.yaml."""
    codename: str = "stable"
    components: list[str] = field(default_factory=lambda: ["main"])
    architectures: list[str] = field(default_factory=lambda: ["amd64", "arm64"])
    label: str = "GitHub Packages"
    origin: str = "github-apt-repo"


class GitHubAPI:
    """GitHub API client for fetching releases."""

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.base_url = "https://api.github.com"

    def _request(self, endpoint: str) -> dict | list:
        """Make an authenticated request to GitHub API."""
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "apt-repo-generator",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        req = Request(url, headers=headers)
        try:
            with urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode())
        except HTTPError as e:
            if e.code == 404:
                raise RuntimeError(f"Repository or release not found: {endpoint}")
            elif e.code == 403:
                raise RuntimeError(
                    f"Rate limit exceeded. Set GITHUB_TOKEN env var for higher limits."
                )
            raise

    def get_repo(self, repo: str) -> dict:
        """Get repository metadata."""
        return self._request(f"repos/{repo}")

    def get_releases(self, repo: str, per_page: int = 30) -> list[dict]:
        """Get releases for a repository."""
        return self._request(f"repos/{repo}/releases?per_page={per_page}")

    def get_latest_release(self, repo: str) -> dict:
        """Get the latest release."""
        return self._request(f"repos/{repo}/releases/latest")


def extract_version(tag: str) -> str:
    """Extract version number from tag (removes 'v' prefix)."""
    return tag.lstrip("vV")


def extract_major_minor(version: str) -> str:
    """Extract major.minor from version string."""
    parts = version.split(".")
    if len(parts) >= 2:
        # Handle versions like "2.64.0" -> "2.64"
        return f"{parts[0]}.{parts[1]}"
    return version


def detect_architecture(filename: str) -> Optional[str]:
    """Detect architecture from .deb filename."""
    filename_lower = filename.lower()
    for arch, patterns in ARCH_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, filename_lower):
                return arch
    return None


def compute_hashes(filepath: Path) -> tuple[str, str, str]:
    """Compute MD5, SHA256, and SHA512 hashes of a file."""
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    sha512 = hashlib.sha512()

    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            md5.update(chunk)
            sha256.update(chunk)
            sha512.update(chunk)

    return md5.hexdigest(), sha256.hexdigest(), sha512.hexdigest()


def download_file(url: str, dest: Path, token: Optional[str] = None) -> None:
    """Download a file from URL to destination."""
    headers = {"User-Agent": "apt-repo-generator"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = Request(url, headers=headers)
    with urlopen(req, timeout=300) as response:
        with open(dest, "wb") as f:
            shutil.copyfileobj(response, f)


def extract_deb_control(deb_path: Path) -> dict:
    """Extract control information from a .deb file."""
    control = {}
    try:
        # Use dpkg-deb if available
        result = subprocess.run(
            ["dpkg-deb", "--info", str(deb_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if ":" in line:
                    key, _, value = line.strip().partition(":")
                    key = key.strip().lower()
                    value = value.strip()
                    if key and value:
                        control[key] = value
    except (subprocess.SubprocessError, FileNotFoundError):
        # dpkg-deb not available, try ar + tar
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmppath = Path(tmpdir)
                # Extract control.tar from .deb
                subprocess.run(
                    ["ar", "x", str(deb_path), "control.tar.gz"],
                    cwd=tmppath,
                    capture_output=True,
                    timeout=30,
                )
                control_tar = tmppath / "control.tar.gz"
                if not control_tar.exists():
                    subprocess.run(
                        ["ar", "x", str(deb_path), "control.tar.xz"],
                        cwd=tmppath,
                        capture_output=True,
                        timeout=30,
                    )
                    control_tar = tmppath / "control.tar.xz"

                if control_tar.exists():
                    subprocess.run(
                        ["tar", "xf", str(control_tar)],
                        cwd=tmppath,
                        capture_output=True,
                        timeout=30,
                    )
                    control_file = tmppath / "control"
                    if control_file.exists():
                        content = control_file.read_text()
                        for line in content.split("\n"):
                            if ":" in line and not line.startswith(" "):
                                key, _, value = line.partition(":")
                                control[key.strip().lower()] = value.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    return control


def find_deb_assets(
    release_data: dict,
    project: Project,
    architectures: list[str],
) -> list[dict]:
    """Find .deb assets in a release matching the project configuration."""
    assets = []

    for asset in release_data.get("assets", []):
        name = asset.get("name", "")

        # Must be a .deb file
        if not name.endswith(".deb"):
            continue

        # Apply custom pattern filter if specified
        if project.asset_pattern:
            if not re.search(project.asset_pattern, name, re.IGNORECASE):
                continue

        # Detect architecture
        arch = detect_architecture(name)
        if arch and arch in architectures:
            assets.append({
                "name": name,
                "url": asset.get("browser_download_url", ""),
                "size": asset.get("size", 0),
                "architecture": arch,
            })

    return assets


def fetch_releases(
    github: GitHubAPI,
    project: Project,
    settings: RepoSettings,
) -> list[Release]:
    """Fetch and process releases for a project."""
    releases_data = github.get_releases(project.repo)

    # Group releases by major.minor version
    releases_by_minor: dict[str, dict] = {}

    for release_data in releases_data:
        # Skip pre-releases and drafts
        if release_data.get("prerelease") or release_data.get("draft"):
            continue

        tag = release_data["tag_name"]
        version = extract_version(tag)
        major_minor = extract_major_minor(version)

        # Keep only the first (latest) release for each major.minor
        if major_minor not in releases_by_minor:
            releases_by_minor[major_minor] = release_data

    # Sort by version (newest first)
    sorted_versions = sorted(
        releases_by_minor.keys(),
        key=lambda v: [int(x) if x.isdigit() else 0 for x in v.split(".")],
        reverse=True,
    )

    # Determine how many versions to keep
    if project.keep_versions > 0:
        versions_to_keep = sorted_versions[: project.keep_versions + 1]
    else:
        versions_to_keep = sorted_versions[:1]  # Only latest

    releases = []
    for major_minor in versions_to_keep:
        release_data = releases_by_minor[major_minor]
        tag = release_data["tag_name"]
        version = extract_version(tag)

        # Find .deb assets
        deb_assets = find_deb_assets(release_data, project, settings.architectures)

        if deb_assets:
            release = Release(
                tag=tag,
                version=version,
                major_minor=major_minor,
            )

            for asset in deb_assets:
                package = DebPackage(
                    name=project.name,
                    version=version,
                    architecture=asset["architecture"],
                    url=asset["url"],
                    filename=asset["name"],
                    size=asset["size"],
                    project_repo=project.repo,
                    description=project.description or f"{project.name} from GitHub",
                    homepage=f"https://github.com/{project.repo}",
                )
                release.packages.append(package)

            releases.append(release)

    return releases


def load_manifest(output_dir: Path) -> dict[str, list[dict]]:
    """Load existing packages manifest."""
    manifest_path = output_dir / MANIFEST_FILE
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {"packages": []}


def save_manifest(output_dir: Path, packages: list[DebPackage]) -> None:
    """Save packages manifest."""
    manifest_path = output_dir / MANIFEST_FILE
    data = {
        "packages": [asdict(pkg) for pkg in packages],
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    with open(manifest_path, "w") as f:
        json.dump(data, f, indent=2)


def generate_packages_file(packages: list[DebPackage], pool_prefix: str) -> str:
    """Generate Packages file content."""
    entries = []

    for pkg in packages:
        entry = f"""Package: {pkg.name}
Version: {pkg.version}
Architecture: {pkg.architecture}
Maintainer: {pkg.maintainer or 'Unknown'}
Installed-Size: {pkg.size // 1024}
Depends: {pkg.depends or ''}
Filename: {pool_prefix}/{pkg.filename}
Size: {pkg.size}
MD5sum: {pkg.md5}
SHA256: {pkg.sha256}
SHA512: {pkg.sha512}
Section: {pkg.section}
Priority: {pkg.priority}
Homepage: {pkg.homepage}
Description: {pkg.description}
"""
        entries.append(entry.strip())

    return "\n\n".join(entries) + "\n" if entries else ""


def generate_release_file(
    settings: RepoSettings,
    packages_files: dict[str, tuple[int, str, str, str]],  # path -> (size, md5, sha256, sha512)
) -> str:
    """Generate Release file content."""
    now = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S UTC")

    content = f"""Origin: {settings.origin}
Label: {settings.label}
Suite: {settings.codename}
Codename: {settings.codename}
Date: {now}
Architectures: {' '.join(settings.architectures)}
Components: {' '.join(settings.components)}
Description: APT repository generated from GitHub releases
"""

    # Add checksums
    md5_lines = []
    sha256_lines = []
    sha512_lines = []

    for filepath, (size, md5, sha256, sha512) in sorted(packages_files.items()):
        md5_lines.append(f" {md5} {size:>8} {filepath}")
        sha256_lines.append(f" {sha256} {size:>8} {filepath}")
        sha512_lines.append(f" {sha512} {size:>8} {filepath}")

    content += "MD5Sum:\n" + "\n".join(md5_lines) + "\n"
    content += "SHA256:\n" + "\n".join(sha256_lines) + "\n"
    content += "SHA512:\n" + "\n".join(sha512_lines) + "\n"

    return content


def sign_release(release_file: Path, gpg_key: Optional[str] = None) -> bool:
    """Sign the Release file with GPG."""
    try:
        # Create detached signature (Release.gpg)
        gpg_cmd = ["gpg", "--batch", "--yes", "-abs"]
        if gpg_key:
            gpg_cmd.extend(["--local-user", gpg_key])
        gpg_cmd.extend(["-o", str(release_file.with_suffix(".gpg")), str(release_file)])

        subprocess.run(gpg_cmd, check=True, capture_output=True, timeout=60)

        # Create inline signature (InRelease)
        inrelease = release_file.parent / "InRelease"
        gpg_cmd = ["gpg", "--batch", "--yes", "--clearsign"]
        if gpg_key:
            gpg_cmd.extend(["--local-user", gpg_key])
        gpg_cmd.extend(["-o", str(inrelease), str(release_file)])

        subprocess.run(gpg_cmd, check=True, capture_output=True, timeout=60)

        return True
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"Warning: GPG signing failed: {e}")
        return False


def export_public_key(output_path: Path, gpg_key: Optional[str] = None) -> bool:
    """Export GPG public key for repository users."""
    try:
        gpg_cmd = ["gpg", "--armor", "--export"]
        if gpg_key:
            gpg_cmd.append(gpg_key)

        result = subprocess.run(gpg_cmd, capture_output=True, timeout=60)
        if result.returncode == 0 and result.stdout:
            output_path.write_bytes(result.stdout)
            return True
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return False


def load_config(config_path: Path) -> tuple[RepoSettings, list[Project]]:
    """Load configuration from projects.yaml."""
    with open(config_path) as f:
        data = yaml.safe_load(f)

    settings_data = data.get("settings", {})
    settings = RepoSettings(
        codename=settings_data.get("codename", "stable"),
        components=settings_data.get("components", ["main"]),
        architectures=settings_data.get("architectures", ["amd64", "arm64"]),
        label=settings_data.get("label", "GitHub Packages"),
        origin=settings_data.get("origin", "github-apt-repo"),
    )

    projects = []
    for proj_data in data.get("projects", []):
        projects.append(Project(
            repo=proj_data["repo"],
            name=proj_data.get("name", ""),
            description=proj_data.get("description", ""),
            keep_versions=proj_data.get("keep_versions", 0),
            asset_pattern=proj_data.get("asset_pattern", ""),
        ))

    return settings, projects


def cleanup_old_packages(
    pool_dir: Path,
    all_packages: list[DebPackage],
) -> list[Path]:
    """Remove .deb files that are no longer in the manifest."""
    removed = []
    current_filenames = {pkg.filename for pkg in all_packages}

    if pool_dir.exists():
        for deb_file in pool_dir.glob("*.deb"):
            if deb_file.name not in current_filenames:
                deb_file.unlink()
                removed.append(deb_file)

    return removed


def main():
    parser = argparse.ArgumentParser(
        description="Generate APT repository from GitHub releases"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("projects.yaml"),
        help="Path to projects.yaml config file",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("repo"),
        help="Output directory for repository (usually gh-pages checkout)",
    )
    parser.add_argument(
        "--project", "-p",
        type=str,
        help="Process only specific project (name or owner/repo)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List configured projects and exit",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done without downloading",
    )
    parser.add_argument(
        "--gpg-key", "-k",
        type=str,
        help="GPG key ID for signing (optional)",
    )
    parser.add_argument(
        "--no-sign",
        action="store_true",
        help="Skip GPG signing",
    )

    args = parser.parse_args()

    # Load configuration
    settings, all_projects = load_config(args.config)

    # List mode
    if args.list:
        print("Configured projects:")
        for proj in all_projects:
            print(f"  - {proj.repo} (name: {proj.name}, keep_versions: {proj.keep_versions})")
        return

    # Determine which projects to process
    if args.project:
        projects_to_process = [
            p for p in all_projects
            if p.name == args.project or p.repo == args.project
        ]
        if not projects_to_process:
            print(f"Error: Project '{args.project}' not found in config")
            return 1
    else:
        projects_to_process = all_projects

    # Initialize GitHub API client
    github = GitHubAPI()

    output_dir = args.output
    pool_dir = output_dir / "pool" / "main"

    # Load existing manifest (for incremental updates)
    manifest = load_manifest(output_dir)
    existing_packages = [
        DebPackage(**pkg_data) for pkg_data in manifest.get("packages", [])
    ]

    # Filter out packages from projects we're about to update
    projects_to_update = {p.repo for p in projects_to_process}
    preserved_packages = [
        pkg for pkg in existing_packages
        if pkg.project_repo not in projects_to_update
    ]

    if not args.dry_run:
        pool_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(projects_to_process)} project(s)...")
    if preserved_packages:
        print(f"Preserving {len(preserved_packages)} package(s) from other projects")

    # Collect new packages
    new_packages: list[DebPackage] = []

    for project in projects_to_process:
        print(f"\n{'='*60}")
        print(f"Project: {project.repo}")
        print(f"{'='*60}")

        try:
            releases = fetch_releases(github, project, settings)

            if not releases:
                print(f"  No releases with .deb packages found")
                continue

            for release in releases:
                print(f"\n  Release: {release.tag} (version {release.version})")

                for pkg in release.packages:
                    print(f"    - {pkg.filename} ({pkg.architecture})")

                    if args.dry_run:
                        continue

                    # Download .deb file
                    deb_path = pool_dir / pkg.filename
                    if not deb_path.exists():
                        print(f"      Downloading...")
                        download_file(pkg.url, deb_path, github.token)
                    else:
                        print(f"      Already exists, skipping download")

                    # Compute hashes
                    pkg.md5, pkg.sha256, pkg.sha512 = compute_hashes(deb_path)
                    pkg.size = deb_path.stat().st_size

                    # Extract control info from .deb
                    control = extract_deb_control(deb_path)
                    if control:
                        pkg.name = control.get("package", pkg.name)
                        pkg.description = control.get("description", pkg.description)
                        pkg.maintainer = control.get("maintainer", pkg.maintainer)
                        pkg.depends = control.get("depends", pkg.depends)
                        pkg.section = control.get("section", pkg.section)
                        pkg.priority = control.get("priority", pkg.priority)
                        pkg.homepage = control.get("homepage", pkg.homepage)

                    new_packages.append(pkg)

        except Exception as e:
            print(f"  Error processing {project.repo}: {e}")
            continue

    if args.dry_run:
        print("\n[Dry run - no files written]")
        return

    # Combine preserved and new packages
    all_packages = preserved_packages + new_packages

    # Clean up old .deb files no longer needed
    removed = cleanup_old_packages(pool_dir, all_packages)
    if removed:
        print(f"\nRemoved {len(removed)} old package(s)")

    # Save manifest
    save_manifest(output_dir, all_packages)

    print(f"\n{'='*60}")
    print("Generating repository metadata...")
    print(f"{'='*60}")

    # Group packages by architecture
    packages_by_arch: dict[str, list[DebPackage]] = {}
    for arch in settings.architectures:
        packages_by_arch[arch] = [
            pkg for pkg in all_packages if pkg.architecture == arch
        ]

    # Create dists structure
    dist_dir = output_dir / "dists" / settings.codename
    packages_files: dict[str, tuple[int, str, str, str]] = {}

    for component in settings.components:
        for arch in settings.architectures:
            arch_dir = dist_dir / component / f"binary-{arch}"
            arch_dir.mkdir(parents=True, exist_ok=True)

            packages = packages_by_arch.get(arch, [])
            packages_content = generate_packages_file(packages, f"pool/{component}")

            # Write Packages file
            packages_file = arch_dir / "Packages"
            packages_file.write_text(packages_content)

            # Write Packages.gz
            packages_gz = arch_dir / "Packages.gz"
            with gzip.open(packages_gz, "wt") as f:
                f.write(packages_content)

            # Calculate hashes for Release file
            for pfile in [packages_file, packages_gz]:
                rel_path = str(pfile.relative_to(dist_dir))
                size = pfile.stat().st_size
                md5, sha256, sha512 = compute_hashes(pfile)
                packages_files[rel_path] = (size, md5, sha256, sha512)

            print(f"  Created {component}/binary-{arch}/Packages ({len(packages)} packages)")

    # Generate Release file
    release_content = generate_release_file(settings, packages_files)
    release_file = dist_dir / "Release"
    release_file.write_text(release_content)
    print(f"  Created Release file")

    # Sign repository
    if not args.no_sign:
        if sign_release(release_file, args.gpg_key):
            print(f"  Created Release.gpg and InRelease (signed)")

            # Export public key
            pubkey_path = output_dir / "public.key"
            if export_public_key(pubkey_path, args.gpg_key):
                print(f"  Exported public key to public.key")
        else:
            print(f"  Skipped signing (GPG not available or no key)")

    print(f"\nRepository generated in: {output_dir}")
    print(f"Total packages: {len(all_packages)}")


if __name__ == "__main__":
    exit(main() or 0)