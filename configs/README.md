# Configuration

Tracked configuration contains only portable defaults and paper facts. Machine-specific paths
belong in `.local/config.yaml` or environment variables. Resolution order is:

1. CLI option
2. environment variable
3. `.local/config.yaml`
4. safe repository-relative default

The dataset registry is an allowlist. The resolver never scans unregistered first-level storage
directories and never follows symlinks by default.

