# ESBMTK v0.14.2.8 Release Materials

This directory contains all the materials prepared for the ESBMTK v0.14.2.8 release.

## Quick Start

To complete the release, follow these steps:

1. **Push the tag** (requires repository write access):
   ```bash
   git push origin v0.14.2.8
   ```

2. **Create GitHub Release**:
   - Visit: https://github.com/uliw/esbmtk/releases/new
   - Select tag: `v0.14.2.8`
   - Title: `ESBMTK v0.14.2.8`
   - Copy content from **GITHUB_RELEASE_BODY.md** into the description
   - Save as draft, review, then publish

## Release Files

### Primary Files (Use These!)

| File | Purpose | Action Required |
|------|---------|----------------|
| **GITHUB_RELEASE_BODY.md** | ⭐ GitHub release description | Copy to GitHub release body |
| **RELEASE_INSTRUCTIONS.md** | Step-by-step completion guide | Follow the instructions |

### Supporting Documentation

| File | Purpose |
|------|---------|
| **RELEASE_NOTES_v0.14.2.8.md** | Comprehensive release notes |
| **RELEASE_CHECKLIST.md** | Complete release process checklist |

## Release Information

- **Version**: 0.14.2.8
- **Release Date**: November 6th, 2025
- **Previous Version**: 0.14.2.7 (May 10th, 2025)
- **Tag**: v0.14.2.8
- **Tagged Commit**: df23d748fa6c332e700fe2b277c8992eff6719a2

## What's in This Release

### Bug Fixes
- Fixed `reverse_time` display for plots with more than one x-axis
- Reservoir mass calculations are now updated once integration finishes
- Enhanced robustness when reading third-party generated CSV data with improved encoding error detection

### New Features
- Gas Exchange Connections now support the `scale` keyword
- Unit-registered output variables (add `_u` suffix to show units)
- `reverse_time` keyword supported by Signal, ExternalData, and plot functions
- Warnings are now written to the log file
- ExternalData accepts the `plot_args` keyword for plot customization

## Requirements

- Python 3.12 or higher (same as 0.14.2.x series)
- No breaking changes from 0.14.2.7

## Next Steps After Release

1. ✅ Tag pushed to GitHub
2. ✅ GitHub release published
3. ⏳ PyPI upload (optional - if doing a PyPI release)
4. ⏳ Verify conda-forge auto-update
5. ⏳ Update documentation (should auto-build)
6. ⏳ Announce release (if applicable)

## Questions?

Refer to **RELEASE_INSTRUCTIONS.md** for detailed guidance on each step.

---

**Status**: Release preparation complete. Manual steps required to push tag and create GitHub release.
