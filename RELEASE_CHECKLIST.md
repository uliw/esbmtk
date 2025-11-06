# Release Checklist for ESBMTK v0.14.2.8

This checklist outlines the steps needed to complete the release of ESBMTK v0.14.2.8.

## Pre-Release Preparation

- [x] Update CHANGELOG.md with v0.14.2.8 changes
- [x] Update CHANGELOG.rst with v0.14.2.8 changes
- [x] Update README.md with new version number and date
- [x] Update README.org with new version number and date
- [x] Create release notes (RELEASE_NOTES_v0.14.2.8.md)
- [x] Create GitHub release body (GITHUB_RELEASE_BODY.md)
- [x] Create git tag v0.14.2.8

## Release Steps

### 1. Push Tag to GitHub
```bash
git push origin v0.14.2.8
```

### 2. Create GitHub Release
1. Go to https://github.com/uliw/esbmtk/releases/new
2. Select tag: `v0.14.2.8`
3. Release title: `ESBMTK v0.14.2.8`
4. Copy content from `GITHUB_RELEASE_BODY.md` into the description
5. **Create as DRAFT** initially to review
6. Once reviewed, publish the release

### 3. Build and Publish to PyPI
```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build distribution packages
python -m build

# Check the build
twine check dist/*

# Upload to Test PyPI first (optional)
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

### 4. Update conda-forge (if applicable)
The conda-forge bot should automatically create a PR when PyPI is updated. Review and merge it.

### 5. Post-Release Tasks
- [ ] Verify installation from PyPI: `pip install esbmtk==0.14.2.8`
- [ ] Verify conda-forge package (once available)
- [ ] Update documentation website (readthedocs should auto-build)
- [ ] Announce release (if applicable):
  - Project mailing list
  - Social media
  - Related communities

## Version Information

- **Version**: 0.14.2.8
- **Release Date**: November 6th, 2025
- **Previous Version**: 0.14.2.7 (May 10th, 2025)
- **Python Requirement**: 3.12+
- **Tag Commit**: df23d748fa6c332e700fe2b277c8992eff6719a2

## Important Files for This Release

- `RELEASE_NOTES_v0.14.2.8.md` - Detailed release notes
- `GITHUB_RELEASE_BODY.md` - Concise release body for GitHub
- `CHANGELOG.md` / `CHANGELOG.rst` - Full project changelog
- Tag: `v0.14.2.8`

## Notes

- This is a maintenance and feature enhancement release
- Requires Python 3.12 (same as 0.14.2.x series)
- No breaking changes from 0.14.2.7
- Focus on bug fixes and new features for plotting and data handling

## Troubleshooting

If the PyPI upload fails:
- Check version number in setup.cfg or pyproject.toml
- Verify setuptools_scm is working correctly
- Ensure the tag is pushed before building

If conda-forge doesn't auto-update:
- Manually create PR at https://github.com/conda-forge/esbmtk-feedstock
