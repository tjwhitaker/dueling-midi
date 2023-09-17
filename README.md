# Machine Learning Piano Pedal

[![Watch the video](./nano-with-fan.jpg)](https://youtu.be/L2GwKnC5NbE?si=J_8Zry4PE6UuxCR9)

Here is the repo for my piano pedal project. All models are built with pytorch. src/main.py contains the code we run to interface with the piano. All model code is stored in src/models.py. Each model has it's own file associated with it that contains it's training and prediction code (src/train*{algo}.py and predict*{algo}.py). There are some timing test scattered throghout some of these files. This needs to be abstracted away into another file in the future for easier analysis to be ran. Regardless, they should be easy to comment in/out as needed.
