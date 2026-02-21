#!/usr/bin/env python3
"""
Model Version Cleanup Script.

Keeps only the latest N versions of each model type.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODEL_VERSIONS_DIR, MAX_MODEL_VERSIONS


def cleanup_old_versions(keep: int = None, dry_run: bool = False):
    """
    Remove old model versions, keeping only the latest N.

    Args:
        keep: Number of versions to keep (default from config)
        dry_run: If True, only print what would be done
    """
    keep = keep or MAX_MODEL_VERSIONS
    registry_path = MODEL_VERSIONS_DIR / "model_registry.json"

    if not registry_path.exists():
        print("No model registry found. Nothing to clean up.")
        return

    with open(registry_path, "r") as f:
        registry = json.load(f)

    total_removed = 0

    for model_type, model_info in registry.get("models", {}).items():
        versions = model_info.get("versions", [])

        if not versions:
            continue

        # Sort by creation date (newest first)
        versions.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        # Identify versions to keep and remove
        versions_to_keep = versions[:keep]
        versions_to_remove = versions[keep:]

        if versions_to_remove:
            print(f"\n{model_type.upper()} Model:")
            print(f"  Keeping {len(versions_to_keep)} versions")
            print(f"  Removing {len(versions_to_remove)} versions")

            for old_version in versions_to_remove:
                old_file = MODEL_VERSIONS_DIR / old_version["filename"]

                if old_file.exists():
                    if dry_run:
                        print(f"    [DRY RUN] Would delete: {old_file.name}")
                    else:
                        old_file.unlink()
                        print(f"    Deleted: {old_file.name}")
                        total_removed += 1
                else:
                    print(f"    Skipped (not found): {old_file.name}")

            # Update registry
            model_info["versions"] = versions_to_keep

    # Save updated registry
    if not dry_run and total_removed > 0:
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)
        print(f"\nRegistry updated. Removed {total_removed} model files.")
    elif dry_run:
        print(f"\n[DRY RUN] Would remove {len(versions_to_remove)} versions per model type.")
    else:
        print("\nNo old versions to remove.")


def main():
    parser = argparse.ArgumentParser(
        description="Clean up old model versions"
    )
    parser.add_argument(
        "--keep", "-k",
        type=int,
        default=MAX_MODEL_VERSIONS,
        help=f"Number of versions to keep (default: {MAX_MODEL_VERSIONS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    print("=" * 50)
    print("MODEL VERSION CLEANUP")
    print("=" * 50)
    print(f"Keeping latest {args.keep} versions of each model type")

    if args.dry_run:
        print("[DRY RUN MODE]")

    cleanup_old_versions(keep=args.keep, dry_run=args.dry_run)

    print("\nCleanup complete.")


if __name__ == "__main__":
    main()
