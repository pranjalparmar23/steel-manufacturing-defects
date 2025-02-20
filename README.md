Link to access the dataset  - https://drive.google.com/drive/folders/1ffM5TnlXm_3YbnoeIfeySQmiWPoJl-1R?usp=sharing
Steel Manufacturing Defects Detection

Repository Structure

models/     # Contains trained models (ResNet50, ResUNet, CNN)
resnet/     # Contains Streamlit files for defect detection
smd/        # Contains website files
venv/       # Virtual environment for dependencies
.gitignore  # Git ignore file
.gitattributes # Git attributes file

Installation and Setup

1. Clone the Repository

git clone <repository-url>
cd <repository-name>

2. Activate Virtual Environment

Windows

./venv/Scripts/activate

Running the Project

1. Run the Website

cd smd
python manage.py server

2. Run the Streamlit Application

Detect Defective Images

cd resnet
streamlit run streamlit.py --server.enableXsrfProtection false

Detect Non-Defective Images

cd resnet
streamlit run streamlit2.py --server.enableXsrfProtection false

Models Used

ResNet50: Used for defect detection in images

ResUNet: Used for segmentation tasks

CNN: Used for classification of images

Features

Web Application: Built with Django, accessible via python manage.py server.

Defect Detection: Using Streamlit (streamlit.py for defective images, streamlit2.py for non-defective images).

Trained Models: Stored in models/.

Virtual Environment: Contains dependencies for the project.

Proposed Approach and Details
![image](https://github.com/user-attachments/assets/595ec6b4-5f1c-449e-9a59-b7c0afc0b686)

Sequence Diagram
![image](https://github.com/user-attachments/assets/e8b3245e-42e7-4447-b77b-fb5c1db4b8a9)

Use Case Diagram
![image](https://github.com/user-attachments/assets/04ca5746-e719-4224-9cd0-b3d4e4a7369d)

