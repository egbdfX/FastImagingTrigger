# Fast Imaging Trigger

We have developed a tLISI-based GPU-accelerated fast imaging trigger for transient detection in radio astronomy. The CPU version of tLISI is displayed on [tLISI](https://github.com/egbdfX/Intensity-sensitive-IQAs). Please see our paper in Section [Reference](https://github.com/egbdfX/FastImagingTrigger/tree/main#reference) for more information.

## User guidance

**Step 1:**
Make sure GCCcore, CUDA, and CFITSIO are avaiable. If you see a warning saying ```/usr/bin/ld.gold: warning: /apps/system/easybuild/software/GCCcore/11.2.0/lib/gcc/x86_64-pc-linux-gnu/11.2.0/crtbegin.o: unknown program property type 0xc0010002 in .note.gnu.property section```, you would need to make sure Python is also available.

**Step 2:**
Run the Makefile by ```make```. Note that this Makefile is written for NVIDIA H100. If you are using other GPU, you would need to make sure the CUDA arch is matching.

**Step 3:**
Run the codes by ```./sharedlibrary_gpu dif1.fits dif2.fits snap1.fits```, where ```dif1.fits``` and ```dif2.fits``` are the two difference images (FITS files), and ```snap1.fits``` is the reference snapshot image (FITS file).

**Step 4:**
The codes will output a FITS file named ```output_tLISI.fits```, which is the output tLISI matrix.

## Reference

**When referencing ITW-SSIM or LISI, please cite our related paper:**

X. Li and W. Armour, "Intensity-Sensitive Similarity Indexes for Image Quality Assessment," 2022 26th International Conference on Pattern Recognition (ICPR), Montreal, QC, Canada, 2022, pp. 1975-1981, doi: 10.1109/ICPR56361.2022.9956093.

**When referencing augLISI, please cite our related paper:**

X. Li, K. Adámek, W. Armour, "Intensity-sensitive quality assessment of extended sources in astronomical images," 2024.

**When referencing tLISI, please cite our related paper:**

X. Li, K. Adámek, W. Armour, "GPU-accelerated fast imaging trigger for source localisation in transient detection," 2024.

## License

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
