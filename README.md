# interplayPFC


The experimental data for the experiments reported in the main figures of [Barbosa & Stein et al. (2020)](https://www.biorxiv.org/content/10.1101/763938v1) can be downloaded from [Data](https://github.com/comptelab/interplayPFC/tree/master/Data) (monkeys) from [an OSF repository](https://osf.io/qa34s/) (human EEG), and from [an OSF repository](https://osf.io/8e9y2) (human TMS).

We include here the Python codes that produce each of the main figures in this article. These custom scripts were developed for Python 2.7 (monkey data, human TMS data, and computational modeling) and for Python 3.7.4 (human EEG data).

# Data 

[Data](https://github.com/comptelab/interplayPFC/tree/master/Data) folder constains raw or minimaly processed data. 

If you need less preprocessed data, please contact:

[:envelope:](mailto:cconstan@wakehealth.edu) Christos Constantinidis, for :monkey: data

[:envelope:](mailto:heike.c.stein@gmail.com), Heike Stein for :girl: EEG data.

# Figure Scripts
Each figure folder contains 3 files:
### plot_figure.py

This file loads preprocessed files from the preprocessed_data and generates the final figure.
Statistics are usually done in this script, but might be done in a previous preprocessing state too (see below).

### preprocessed_data/
 
This folder contains all the preprocessed data used by the plot_figure.py file.

### preprocess_scripts/

This folder contains all the scripts that generate preprocessed data from the raw data in Data/ folder.
Preprocessed data is saved in the preprocessed_data folder.

# Models
[serial_bias_models](https://github.com/comptelab/interplayPFC/tree/master/serial_bias_models) contains 2 versions of the serial bias model. 1 with untuned inhibition and another with tuned inhibition ([see Fig S4](https://www.biorxiv.org/content/biorxiv/early/2019/09/12/763938/DC1/embed/media-1.pdf)).

# Contact
If you have any questions, please write to the corresponding author, Albert Comte [:envelope:](mailto:ACOMPTE@clinic.cat), to Joao Barbosa, [:envelope:](mailto:palerma@gmail.com, monkey & TMS code), or Heike Stein (heike.c.stein@gmail.com, human EEG code)
