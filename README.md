# Inertial-Signals-Classifier
Function programmed in MATLAB using the SVM method, which allows to classify actions according to inertial signal data.

The database in the traindata.mat[1] file contains the variables described below, each of which has 7352 rows. Each row corresponds to the signals of an experiment, associated with a specific activity. All signals have been sampled at 50Hz.

Acc_X, Acc_Y, and Acc_Z refer to accelerometer measurements in the X, Y, and Z directions, respectively.

AccBody_X, AccBody_Y, AccBody_Z, refer to the accelerometer measurements in the X, Y and Z directions, after subtracting the component of acceleration due to gravity.

Gyr_X, Gyr_Y, and Gyr_Z refer to gyroscope measurements in the X, Y, and Z directions.

activity, it is a cell array with the name of the corresponding activity.

ID corresponds to a unique identifier for each participant of the experiments. In other words, the measures associated with the same ID correspond to the same person.

[1] traindata.mat : https://drive.google.com/file/d/1iZWJHgLkF1VXBV1lD5Ai7-vtCxmhGn1I/view?usp=sharing

Reference : https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2013-84.pdf
