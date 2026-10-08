# Fake News Detection - Localhost Version

This converts the Colab project into a local Flask web application.

## Folder structure

Fake_News_Localhost/
│
├── app.py
├── Fake.csv
├── True.csv
├── requirements.txt
├── templates/
│   └── index.html
└── static/
    └── style.css

## Setup in VS Code

1. Put `Fake.csv` and `True.csv` in the same folder as `app.py`.

2. Open the folder in VS Code.

3. Open Terminal and run:

   pip install -r requirements.txt

4. Start the application:

   python app.py

5. Open Chrome and go to:

   http://127.0.0.1:5000

The model is trained when `app.py` starts, using the same preprocessing, TF-IDF settings, train/test split and Multinomial Naive Bayes model as the notebook.

The UI is a normal browser website, not an ipywidgets notebook interface.
