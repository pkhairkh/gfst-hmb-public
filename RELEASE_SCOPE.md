# Release Scope

This staging bundle was prepared for public upload with a conservative file policy.

Included:

- Selected source files.
- Selected tests.
- Public release documentation.
- Packaging metadata.

Excluded:

- All `.pt`, `.pth`, `.ckpt`, and `.bin` files.
- Upstream model weights and tokenizer assets.
- Raw reports directory and report zip archives.
- Existing configs with machine-local paths.
- Internal remote execution and infrastructure scripts.
- Junk and cache files.

The source tree in the working repository was not edited to create this release. Public-facing materials were added only inside `_public_release/`.
