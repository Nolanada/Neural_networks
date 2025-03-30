This repository is for the prediction of traffic signs

A. Observation
-After training the first model model0.h4, the accuracy was 0.96 and the prediction_sign(which was called recognition_gui.py) could predict and display just the class number and accuracy. A terminal version of the gui(recognition.py) was also implemented

-After modifying the traffic.py so as to include the class name, a second and third model best_model.h5 and best_model2.h5 were train with the latter having an accuracy of 0.52%. The prediction_sign.py could get 2/5 images from the test images correctly and 8/10 from the  correctly from the dataset

--After training a new model, the accuracy increased to 0.69% and the prediction_sign.py could get 4/6 images correctly. I concluded that the more the model is trained, the more accurate the guess get
