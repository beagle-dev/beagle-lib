BEAGLE is a high-performance library that can perform the core calculations at the heart of most Bayesian and Maximum Likelihood phylogenetics packages. It can make use of highly-parallel processors such as those in graphics cards (GPUs) found in many PCs.

The project involves an open API and fast implementations of a library for evaluating phylogenetic likelihoods (continuous time Markov processes) of biomolecular sequence evolution.

The aim is to provide high performance evaluation 'services' to a wide range of phylogenetic software, both Bayesian samplers and Maximum Likelihood optimizers. This allows these packages to make use of implementations that make use of optimized hardware such as graphics processing units.

Currently the following software packages can make use of the BEAGLE library:
- BEAST http://beast.bio.ed.ac.uk/
- Garli https://garli.googlecode.com/
- MrBayes http://mrbayes.sourceforge.net/
- PhyML https://phyml.googlecode.com/

A manuscript describes the BEAGLE API and library:  
http://sysbio.oxfordjournals.org/content/61/1/170

The paper describing the algorithms used for calculating likelihoods of sequences on trees using many core devices like graphics processing units (GPUs) is available from:  
http://tree.bio.ed.ac.uk/publications/390/  

Binary installers:
- [BEAGLE v2.1.2 for Mac OS X 10.6 and later](https://www.dropbox.com/s/11kgt2jlq3lzln3/BEAGLE-2.1.2.pkg)
- [BEAGLE v2.1 for Windows XP and later](https://www.dropbox.com/s/61z48jvruzkwkku/BEAGLE-2.1.msi)

Installation instructions:
- [Instructions for installing BEAGLE on Mac OS X](https://github.com/beagle-dev/beagle-lib/wiki/MacInstallInstructions)
- [Instructions for installing BEAGLE on Windows](https://github.com/beagle-dev/beagle-lib/wiki/WindowsInstallInstructions)
- [Instructions for installing BEAGLE on Linux](https://github.com/beagle-dev/beagle-lib/wiki/LinuxInstallInstructions) 

Documentation:
- [API documentation](https://beagle-dev.github.io/html/beagle_8h.html)
- [Release notes](https://github.com/beagle-dev/beagle-lib/wiki/ReleaseNotes)

# Acknowledgements
- This project is supported in part through the National Science Foundation grants IIS 1251151 and DMS 1264153.
