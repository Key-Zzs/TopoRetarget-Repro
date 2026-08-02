"""Optional Stage-16 Isaac Lab asset boundary.

This package contains only simulator-independent contracts at import time.
Isaac Sim and Isaac Lab are imported lazily by the runtime scripts.
"""

from .asset_contracts import (
    AssetMigrationConfig,
    BoundedAssetRecovery,
    HOCapObjectSpec,
    WujiAssetSpec,
    load_asset_migration_config,
)

__all__ = [
    "AssetMigrationConfig",
    "BoundedAssetRecovery",
    "HOCapObjectSpec",
    "WujiAssetSpec",
    "load_asset_migration_config",
]
