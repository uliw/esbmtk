# Instructions for Completing the v0.14.2.8 Release

## Current Status

The release preparation for ESBMTK v0.14.2.8 has been completed with the following:

✅ **Completed:**
- Release notes created (RELEASE_NOTES_v0.14.2.8.md)
- GitHub release body created (GITHUB_RELEASE_BODY.md)
- Release checklist created (RELEASE_CHECKLIST.md)
- Git tag `v0.14.2.8` created locally on commit df23d74

⚠️ **Pending Manual Steps:**
The tag has been created locally but needs to be pushed to GitHub. This must be done manually.

## Step 1: Push the Tag to GitHub

From the repository root, run:

```bash
git push origin v0.14.2.8
```

This will push the annotated tag to GitHub, making it available for creating a release.

## Step 2: Create GitHub Release Draft

1. **Navigate to GitHub Releases:**
   - Go to https://github.com/uliw/esbmtk/releases/new
   - Or from the repository page, click "Releases" → "Draft a new release"

2. **Configure the Release:**
   - **Choose a tag:** Select `v0.14.2.8` from the dropdown
   - **Release title:** `ESBMTK v0.14.2.8`
   - **Description:** Copy the content from `GITHUB_RELEASE_BODY.md`

3. **Create as Draft:**
   - Check the "Set as a pre-release" box if this is a pre-release
   - Click "Save draft" to create as a draft first
   - Review the draft before publishing

4. **Publish:**
   - Once everything looks good, click "Publish release"

## Step 3: Verify the Release

After publishing, verify:
- Release appears at https://github.com/uliw/esbmtk/releases
- Tag is visible at https://github.com/uliw/esbmtk/tags
- Release notes are formatted correctly

## Step 4: Build and Publish to PyPI (Optional)

If you want to publish to PyPI:

```bash
# Ensure you're on the tagged commit
git checkout v0.14.2.8

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Install build tools if needed
pip install build twine

# Build distribution packages
python -m build

# Check the build
twine check dist/*

# Upload to PyPI (requires PyPI credentials)
twine upload dist/*
```

## Alternative: Using GitHub CLI

If you have GitHub CLI installed, you can create the release directly:

```bash
# Push the tag first
git push origin v0.14.2.8

# Create a draft release
gh release create v0.14.2.8 \
  --title "ESBMTK v0.14.2.8" \
  --notes-file GITHUB_RELEASE_BODY.md \
  --draft

# Or publish directly (without --draft flag)
gh release create v0.14.2.8 \
  --title "ESBMTK v0.14.2.8" \
  --notes-file GITHUB_RELEASE_BODY.md
```

## Files Reference

- **RELEASE_NOTES_v0.14.2.8.md** - Comprehensive release notes with all details
- **GITHUB_RELEASE_BODY.md** - Concise version for GitHub release body (use this!)
- **RELEASE_CHECKLIST.md** - Complete checklist for release process
- **CHANGELOG.md / CHANGELOG.rst** - Already updated with v0.14.2.8 changes

## Tag Information

- **Tag name:** v0.14.2.8
- **Tagged commit:** df23d748fa6c332e700fe2b277c8992eff6719a2
- **Commit message:** "docs: update README files with latest ESBMTK version information"
- **Tag annotation:** Contains release summary

## Questions or Issues?

If you encounter any issues:
1. Check that the tag exists locally: `git tag -l | grep v0.14.2.8`
2. Verify tag details: `git show v0.14.2.8 --no-patch`
3. Ensure you have push permissions to the repository
4. Make sure you're authenticated with GitHub (token or SSH)

---

**Note:** The release draft preparation is complete. The final steps (pushing tag and creating GitHub release) require manual execution with appropriate repository permissions.
