# Random Crop Registration
Detect if a pair of images are properly registered through the computation of optical flow. 

<sub><sup>* This is part of my contribution to a Capstone Research Project involving the generation of consistent 360-degree panoramas from unregistered/incompatible images.</sup></sub><br/>
<sub><sup>** A portion of this project placed second in Biomedical Engineering/third in Robotics & Intelligent at the Central Kentucky Regional Science Fair.</sup></sub>

## How to Use
1. `initialize.py` defines a model instance that is untrained (if no model exists at `model/checkpoint.txt`). By default, the model is trained for 75 epochs on the [Skyfinder Dataset](https://cs.valdosta.edu/~rpmihail/skyfinder/).
2. Evaluation is accuracy-based. The entire training dataset is registered, and as such automatically categorizes the network's performance. Due to the testing dataset's inclusion of both registered/unregistered pairs, user input is currently required for the measurement of accuracy.
3. `P Accuracy` is defined as "proper accuracy", and measures the accuracy of the network on image pairs without data truncation.
