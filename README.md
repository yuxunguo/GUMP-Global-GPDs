# GUMP: Generalized Parton Distributions through Universal Moment Parameterization

This repository contains the source code for the **Generalized Parton Distributions through Universal Moment Parameterization (GUMP)** program, designed for global analysis of GPDs.

## Overview

- **GPD Calculations:**  
  The script under folder `/Examples` provides example workflows to generate PDFs, GPDs and cross-sections using GUMP parameters obtained from the fitting program.

- **Performance Note:**  
  Evaluating GPDs can be computationally intensive, as it involves QCD evolution in moment space and numerical contour integrals to transform back to the momentum fraction (x) space.

## Usage

- **For simple usage** install through pip by ```pip install gumpgpd```. Check the script under ```/Examples``` for example usage.

- **For modified package and developer**  download the whole branch and run ```pip install -e .``` to install locally (Make sure to ```pip uninstall gumpgpd``` to avoid conflicts if you install the pip version already)

## Documentation
For detailed documentation and further information, please visit:  
[https://yuxunguo.github.io/GUMP-Global-GPDs/](https://yuxunguo.github.io/GUMP-Global-GPDs/)
