# Convolution Neural Network for Aeromagnetic Interpretation of Lineaments
This project holds the trained CNN model used in the paper:

T. Naprstek and R.S. Smith, 2021, Convolution Neural Networks Applied to the Interpretation of Lineaments in Aeromagnetic Data, Geophysics, (preprint).
https://library.seg.org/doi/abs/10.1190/geo2020-0779.1

The CNN, NaprstekSmith_CNN_v1.h5, was developed using TensorFlow 2.0. The python script (which requires the TensorFlow 2.0 library) will load in the trained CNN model and apply it to a new aeromagnetic dataset using its moving-window approach. For an example of its usage, it can be run and applied to the test_model.csv dataset. This test model dataset was created using GRAV_MAG_PRISM (de Barros, A., S. Bongiolo, J. de Souza, F. J. F. Ferreira, and L. G. de Castro, 2013, Grav mag prism: a matlab/octave program to generate gravity and magnetic anomalies due to rectangular prismatic bodies, Brazilian Journal of Geophysics, 31, 347-363.)
