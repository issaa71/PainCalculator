# Deploying the Pain Calculator Online

This guide provides multiple options for deploying the Pain Calculator web application online to make it easily shareable.

## Option 1: Streamlit Sharing (Easiest)

Streamlit offers a free hosting service called Streamlit Community Cloud that makes it simple to deploy Streamlit apps.

### Prerequisites:
- A GitHub account
- Your code pushed to a GitHub repository

### Steps:

1. **Create a GitHub repository**:
   - Create a new repository on GitHub
   - Upload the following files to the repository:
     - `pain_calculator.py`
     - `pain_calculator_web.py`
     - `trained_models/` directory with your pre-trained models
     - `requirements.txt` (create this file with the dependencies, see below)

2. **Create requirements.txt**:
   - Create a file named `requirements.txt` with the following content:
   ```
   streamlit>=1.15.0
   pandas>=1.3.0
   numpy>=1.20.0
   scikit-learn>=1.0.0
   matplotlib>=3.4.0
   ```

3. **Modify pain_calculator_web.py** (if needed):
   - Ensure all file paths use relative paths
   - Ensure you have appropriate error handling if models aren't found

4. **Deploy on Streamlit Sharing**:
   - Go to [Streamlit Sharing](https://streamlit.io/sharing)
   - Log in with your GitHub account
   - Click "New app"
   - Select your repository, branch, and the path to `pain_calculator_web.py`
   - Click "Deploy"

5. **Share the URL**:
   - Streamlit will provide you with a URL (something like `https://share.streamlit.io/username/repo/pain_calculator_web.py`)
   - Share this URL with anyone who needs to use the calculator

## Option 2: PythonAnywhere (More Control)

PythonAnywhere offers a free tier that can host your Streamlit application with more control.

### Prerequisites:
- A PythonAnywhere account (free tier works)

### Steps:

1. **Create a PythonAnywhere account**:
   - Go to [PythonAnywhere](https://www.pythonanywhere.com/) and sign up for a free account

2. **Upload your files**:
   - From the Dashboard, go to "Files" tab
   - Create a new directory for your project (e.g., `pain_calculator`)
   - Upload the following files:
     - `pain_calculator.py`
     - `pain_calculator_web.py`
     - `requirements.txt` (as created above)
     - Your pre-trained model files

3. **Open a Bash console**:
   - From the Dashboard, go to "Consoles" tab
   - Click "Bash" to open a new Bash console

4. **Set up a virtual environment and install dependencies**:
   ```bash
   cd pain_calculator
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

5. **Create a start script** (create a file named `start_app.sh`):
   ```bash
   #!/bin/bash
   cd ~/pain_calculator
   source venv/bin/activate
   streamlit run pain_calculator_web.py --server.port 8501 --server.address 0.0.0.0
   ```

6. **Make the script executable**:
   ```bash
   chmod +x start_app.sh
   ```

7. **Set up a custom Web app**:
   - Go to the "Web" tab on PythonAnywhere
   - Add a new web app
   - Choose "Manual configuration" and select Python version
   - In the "Code" section, set the path to your project directory
   - In the "WSGI configuration file", modify it to point to your Streamlit app
   - Set up a custom startup script in the "Virtualenv" section

8. **Set up a scheduled task**:
   - Go to "Tasks" tab
   - Add a scheduled task to run your `start_app.sh` script

9. **Share your PythonAnywhere URL**:
   - Your app will be available at `yourusername.pythonanywhere.com`

## Option 3: Heroku (Advanced)

Heroku is a powerful platform for deploying web applications.

### Prerequisites:
- A Heroku account
- Git installed on your computer
- Heroku CLI installed

### Steps:

1. **Create a new Heroku app**:
   ```bash
   heroku login
   heroku create your-pain-calculator-app
   ```

2. **Create necessary files**:
   - `requirements.txt`: List of dependencies
   - `Procfile`: Contains the command to run your app
   - `setup.sh`: Script to set up Streamlit

3. **Create Procfile** (with this content):
   ```
   web: sh setup.sh && streamlit run pain_calculator_web.py
   ```

4. **Create setup.sh**:
   ```bash
   mkdir -p ~/.streamlit/
   echo "[server]
   headless = true
   port = $PORT
   enableCORS = false
   " > ~/.streamlit/config.toml
   ```

5. **Modify your code to use environment variables** if needed.

6. **Deploy to Heroku**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git push heroku master
   ```

7. **Share your Heroku URL**:
   - Your app will be available at `https://your-pain-calculator-app.herokuapp.com`

## Option 4: GitHub Pages with Static Export (Alternative Approach)

If you want a simpler solution and don't need real-time calculations, you could convert your calculator to a static HTML form:

1. Pre-compute results for common inputs
2. Use JavaScript to look up the closest match
3. Deploy as static HTML on GitHub Pages

This approach is more work to implement but is free and highly reliable.

## Important Security and Privacy Considerations

Before deploying your application publicly:

1. **Data Privacy**:
   - Ensure you're not collecting or storing protected health information (PHI)
   - Add appropriate disclaimers about data usage

2. **Medical Disclaimer**:
   - Clearly state that the calculator is for informational purposes only
   - Advise users to consult healthcare professionals for medical advice

3. **Code Security**:
   - Remove any sensitive information from your code
   - Ensure you're not exposing any private credentials

4. **Model Security**:
   - Consider if your models contain any proprietary or sensitive information
   - If needed, implement user authentication for access

## Resources and Further Reading

- [Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app)
- [PythonAnywhere Web App Guide](https://help.pythonanywhere.com/pages/WebAppBasics/)
- [Heroku Python Deployment](https://devcenter.heroku.com/articles/getting-started-with-python)
