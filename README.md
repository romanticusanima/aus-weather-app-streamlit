# Aus Weather App Streamlit

This project demonstrates how to deploy a machine learning model using Streamlit to predict weather in Australia. The application allows users to input today's weather characteristics and get a tomorrow prediction based on the trained model Random Forest.

You can test the application at the link: https://aus-weather-demo-25.streamlit.app/

If the link displays the message "This app has gone to sleep due to inactivity. Would you like to wake it back up?" simply click the "Yes, get this app back up!" button and wait half a minute.

## Project structure

- **data/**: Folder which contains dataset (`weatherAUS.csv`).
- **images/**: Folder for storing images used in the application.
- **model/**: Folder containing the trained ML model.
- **utils/**: Folder containing helper functions.
- **app.py**: Main application file Streamlit.
- **requirements.txt**: List of required Python packages.
- **train.ipynb**: Jupyter Notebook for training Random Forest model.

### Requirements

You will need all packages specified in `requirements.txt`.

### Installation

1. **Clone repository**:
   ```bash
   git clone https://github.com/romanticusanima/aus-weather-app-streamlit.git
   cd aus-weather-app-streamlit
   ```

2. **Create virtual environment** (optional):
   ```bash
   python -m venv venv
   source venv/bin/activate   # with Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

### Train model

To train the Random Forest classifier, run the Jupyter Notebook `train.ipynb`:

1. Open Jupyter Notebook `train.ipynb`.

2. Follow the steps described in the notebook to train the model and save it to the directory `model/`.

### Running the application Streamlit

Run the Streamlit application locally using the following command:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.
