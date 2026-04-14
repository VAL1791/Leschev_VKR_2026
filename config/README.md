# Local Paths Config

Create `config/local_paths.toml` from `config/local_paths.example.toml` when raw datasets live outside the repo defaults.

The notebook-first flow uses this file to resolve:

- the raw-data root;
- the `dashboard/artifacts/` export folder;
- per-dataset overrides for local storage.

The real config file is intentionally ignored by git.
