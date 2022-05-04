# CDDD

- download data from https://zinc.docking.org/tranches/home/
    - use `wget -i <file_name>` to get .smi files
    - each .smi file has the notation and some number (this number turned out to be the ZINC15 identifier)

- `conda install -c rdkit rdkit` to install rdkit.

- Use AAAA.csv as sample data, small number of molecules, runs on CPU.

- `python cddd.py` to run the code.

## Additional information

- To run on full data, obtain all .smi files, and concatenate into one. Get rid of all the header lines.
- Preferably run on GPU.

## Todo

- Bash script to create .csv dataset
- Script to create molecular properties dataset
- Regeneration of molecules from decoder output
- Improvements on fully-connected network for effective translation.
    
