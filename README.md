# CAR-Net
This is an official implementation of CarNET, entitled Lightweight ECG-Guided Knowledge Distillation for Cardio-Informed Radar Signal Reconstruction (CAR-Net).

This work resulted in two models: SED-Net (Squeeze and Excitation assisted stacked Deformable convolutional Neural Network), and ECG-HR Regression model (ECG_hr_map). CAR-Net (Cardio-Informed Radar Signal Reconstruction Network), multi-task network for cardiac signal reconstruction, and Heart Rate estimation Simultaneously.

To run the implementation, a few python libraries will be required, which are mentioned in Requirements.txt file.

There is only one script made available for this time. The training functionality for this work, and signal preprocessing pipeline will be made publicly available soon.

The project is organised as follows:
1. Directory *signal_frags* includes a sample from the subject consisting of high quality signal ECG-radar aligned fragments. The whole dataset can be requested by making an email request through ankit.gupta@vsb.cz.
2. *trained_models* contains trained models named as teacher_best.pt (SED-Net) and student_best.pt (CAR-Net).
3. *test.py* runs the data from signal_frags, and perform reconstruction and HR estimations over all signal fragments (10 seconds in length).




In case of any difficulty in running this project, please contact us by email using ankit.gupta@vsb.cz. 
