# Deploying a Machine Learning Model on Heroku with FastAPI

This project contains the development of a classification model on Census Bureau data. The main goal is to robustly deploy a machine learning model into production.

The project follows these steps:
* Develop a machine learning model for a classification task that predicts the salary group of individuals based on 14 different characteristics. The model uses a threshold salary of $50,000. For more information on the dataset used and details about the model, you can refer to the corresponding modelCard.
* Expose the model for inference using a FastAPI app
* Deploy the app using Heroku to provide inference endpoint
* Create a workflow for Continuous Integration / Continuous Deployment using GitHub Actions and Heroku integration with GitHub. The application will only be deployed if the tests, which are integrated and automated, are validated by GitHub Actions upon any modifications made to the codebase.


## Environment Set up

* Download and install conda if you don’t have it already.
    * Use the supplied requirements file to create a new environment, or
    * conda create -n [envname] "python=3.8" scikit-learn dvc pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge
    * Install git either through conda (“conda install git”) or through your CLI, e.g. sudo apt-get git.

### Repositories 

* Create a directory for the project and initialize git.
    * As you work on the code, continually commit changes. Trained models you want to use in production must be committed to GitHub.
* Connect your local git repo to GitHub.
* Setup GitHub Actions on your repo. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
    * Make sure you set up the GitHub Action to have the same version of Python as you used in development.

## GitHub Actions

* Setup GitHub Actions on your repository. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
   * Make sure you set up the GitHub Action to have the same version of Python as you used in development.

## Data

* Download census.csv from the data folder.
* This data is messy, try to open it in pandas and see what you get.
* To clean it, use your favorite text editor to remove all spaces.

## Model

* To train the model run:
```python src/train_model.py```
* or run the entire ML pipeline which starts a local server where you can test the model
```python main.py```


## API Creation

* Create a RESTful API using FastAPI this must implement:
   * GET on the root giving a welcome message.
   * POST that does model inference.
   * Type hinting must be used.
   * Use a Pydantic model to ingest the body from POST. This model should contain an example.
    * Hint: the data has names with hyphens and Python does not allow those as variable names. Do not modify the column names in the csv and instead use the functionality of FastAPI/Pydantic/etc to deal with this.
* Write 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).

## API Deployment

* Create a free Heroku account (for the next steps you can either use the web GUI or download the Heroku CLI).
* Create a new app and have it deployed from your GitHub repository.
   * Enable automatic deployments that only deploy if your continuous integration passes.
   * Hint: think about how paths will differ in your local environment vs. on Heroku.
   * Hint: development in Python is fast! But how fast you can iterate slows down if you rely on your CI/CD to fail before fixing an issue. I like to run flake8 locally before I commit changes.
* Write a script that uses the requests module to do one POST on your live API.
