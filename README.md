# interplayPFC


The experimental data for the experiments reported in the main figures of [Barbosa & Stein et al. (2020)](https://www.biorxiv.org/content/10.1101/763938v1) can be downloaded from [Data] (https://github.com/comptelab/interplayPFC/tree/master/Data) (monkeys) from [an OSF repository](https://osf.io/qa34s/) (for human EEG), and from [an OSF repository](https://osf.io/8e9y2) (for human TMS).

We include here the Python codes that produce each of the main figures in this article. These custom scripts were developed for Python 2.7 (monkey data, human TMS data, and computational modeling) and for Python 3.7.4 (human EEG data).

# Data & Scripts
Each figure folder contains 3 files:
- plot_figure.py:
        this file loads preprocessed files from the preprocessed_data and generates the final figure.
        Statistics are usually done in this script, but might be done in a previous preprocessing state too (see below).

- preprocessed_data/
        this folder contains all the preprocessed data used by the plot_figure.py file.

- preprocess_scripts/
        this folder contains all the scripts that generate preprocessed data from the raw data in Data/ folder.
        Preprocessed data is saved in the preprocessed_data folder.

Data/ constains the raw or minimaly processed data.

serial_bias_models/ contains 2 versions of the serial bias model. 1 with untuned inhibition and another with tuned inhibition.

helpers/ miscellaneous scripts (circular statistics, bootstrap test, etc)

# Contact
If you have any questions, please write to the corresponding author, Albert Comte (ACOMPTE@clinic.cat), to Joao Barbosa, (palerma@gmail.com, monkey & TMS code), or Heike Stein (heike.c.stein@gmail.com, human EEG code)
