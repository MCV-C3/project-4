import wandb
from typing import Any, Dict, Tuple


def _set_by_dotted_path(
    cfg: Dict[str, Any],
    key_path: str,
    value: Any,
    *,
    create_missing: bool = True,
) -> Tuple[bool, Any]:
    """Set a nested config value using a dotted key path.

    Args:
        cfg: Base config dictionary to modify in-place.
        key_path: Dotted path like "training.learning_rate" or "model.params.dropout".
        value: Value to set at the target path.
        create_missing: If True, create intermediate dictionaries as needed.

    Returns:
        A tuple (updated, old_value) where:
          - updated is True if the value was set, False otherwise.
          - old_value is the previous value if it existed, else a sentinel string.
    """
    parts = key_path.split(".")
    cur: Any = cfg

    # Walk all parts except the last
    for i, part in enumerate(parts[:-1]):
        if not isinstance(cur, dict):
            # Can't traverse further; would require overwriting a non-dict.
            return False, f"Blocked at '{'.'.join(parts[:i])}' (not a dict)"

        if part not in cur:
            if not create_missing:
                return False, f"Missing key '{'.'.join(parts[:i+1])}'"
            cur[part] = {}

        elif not isinstance(cur[part], dict):
            # Existing intermediate value is not a dict -> can't go deeper safely.
            return False, f"Blocked at '{'.'.join(parts[:i+1])}' (not a dict)"

        cur = cur[part]

    # Set the final key
    if not isinstance(cur, dict):
        return False, f"Blocked at parent of '{key_path}' (not a dict)"

    leaf = parts[-1]
    old = cur.get(leaf, "New Key")
    cur[leaf] = value
    return True, old


def configure_sweep(cfg: Dict[str, Any], *, create_missing: bool = True) -> Dict[str, Any]:
    """Override cfg values with WandB sweep parameters (supports deep dotted paths).

    Args:
        cfg: Base configuration dictionary to override (modified in-place).
        create_missing: Whether to create missing nested dictionaries for new paths.

    Returns:
        The updated config dictionary (same object as input).
    """
    print("\n>>> Sweep detected! Overriding config with sweep parameters:")

    for key, value in wandb.config.items():
        # If you want to ignore W&B metadata keys, filter them here if needed.
        # e.g., if key.startswith("_"): continue

        if "." in key:
            updated, old = _set_by_dotted_path(
                cfg, key, value, create_missing=create_missing)
            if updated:
                print(f"    Overriding {key}: {old} -> {value}")
            else:
                print(f"    Warning: Could not set {key}: {old}")
        else:
            # Flat keys: either update if exists, or set if create_missing=True
            if key in cfg or create_missing:
                old = cfg.get(key, "New Key")
                cfg[key] = value
                print(f"    Overriding {key}: {old} -> {value}")
            else:
                # leave as-is
                pass

    print("-" * 50)
    return cfg
