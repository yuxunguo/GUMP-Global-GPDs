# GUMP: Generalized Parton Distributions through Universal Moment Parameterization

This repository contains the source code for **GUMP** (Generalized Parton Distributions through Universal Moment Parameterization), a program for global analysis of Generalized Parton Distributions (GPDs).

## Overview

- **GPD Calculations:**  
  The `/Examples` folder contains example scripts that demonstrate how to compute PDFs, GPDs, and cross-sections using GUMP fit parameters.

- **Performance Note:**  
  GPD evaluation is computationally intensive: it requires QCD evolution in Mellin (moment) space followed by a numerical contour integral to transform results back to momentum-fraction ($x$) space.

## Installation

Three installation methods are available, depending on your use case:

- **Recommended — install from GitHub:**  
  Stable release:
  ```
  pip install git+https://github.com/yuxunguo/GUMP-Global-GPDs.git@GUMP1.0
  ```
  Development version:
  ```
  pip install git+https://github.com/yuxunguo/GUMP-Global-GPDs.git@GUMPdev
  ```

- **Quick start — install from PyPI:**
  ```
  pip install gumpgpd
  ```

- **Developer install — editable local install:**  
  Clone the repository, then run the following from the root directory (the parent of `/src`):
  ```
  pip install -e .
  ```
  > If another version of `gumpgpd` is already installed, run `pip uninstall gumpgpd` first to avoid conflicts.

## Usage

After installation, refer to the example scripts in `/Examples` for typical workflows.


## Changelog (Development version)

- **V1.0.1** Fixed an issue where unwanted function calls were triggered during GPD calculations.
- **V1.0.2** Improved caching behavior and user-facing messages.
- **V1.0.3** Introduced hybrid caching with `diskcache`; minor code restructuring.

## Documentation

Full documentation is available at:  
[https://yuxunguo.github.io/GUMP-Global-GPDs/](https://yuxunguo.github.io/GUMP-Global-GPDs/)
