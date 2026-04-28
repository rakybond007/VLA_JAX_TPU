# Third-Party References

This directory is reserved for local-only reference clones related to:

- Qwen3-VL / Qwen3.5 implementations
- JAX / Keras / TPU examples for large VLMs
- Checkpoint conversion utilities

Reference repositories were used during development, but they are not vendored in this repository. The `.gitignore` keeps `third_party/*` ignored except for this README so the public repo stays lightweight and avoids redistributing external source trees.

Typical local layout:

```text
third_party/
├── keras-hub/
├── maxtext/
├── Qwen3-VL/
├── Qwen3-VL-JAX/
└── transformers/
```
