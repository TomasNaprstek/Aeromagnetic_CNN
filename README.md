# Convolution Neural Network for Aeromagnetic Interpretation of Lineaments
This project holds the trained CNN model used in the paper:

Tomas Naprstek and Richard S. Smith, (2022), "Convolutional neural networks applied to the interpretation of lineaments in aeromagnetic data," GEOPHYSICS 87: K1-K13. https://doi.org/10.1190/geo2020-0779.1

The CNN, NaprstekSmith_CNN_v1.h5, was developed using TensorFlow 2.0. The python script (which requires the TensorFlow 2.0 library) will load in the trained CNN model and apply it to a new aeromagnetic dataset using its moving-window approach. For an example of its usage, it can be run and applied to the test_model.csv dataset. This test model dataset was created using GRAV_MAG_PRISM (de Barros, A., S. Bongiolo, J. de Souza, F. J. F. Ferreira, and L. G. de Castro, 2013, Grav mag prism: a matlab/octave program to generate gravity and magnetic anomalies due to rectangular prismatic bodies, Brazilian Journal of Geophysics, 31, 347-363.)
