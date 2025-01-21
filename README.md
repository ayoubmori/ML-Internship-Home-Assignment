# ML-Internship-Home-Assignment

## Requirements
- Python 3.9 or higher.

#### - Install Poetry on your global Python setup
Follow the official [Poetry installation guide](https://python-poetry.org/docs/#installation) to install Poetry on your system.

 - Install requirements
```bash
    cd data-ml-home-assignment
    
    poetry install
```
- Activate the virtual environment

```bash
    poetry shell
```
#### - Start the application
```sh
    sh run.sh
```
- API : http://localhost:8000
- Streamlit Dashboard : http://localhost:9000

P.S You can check the log files for any improbable issues with your execution.
## Before we begin
- In this assignement, you will be asked to write, refactor, and test code. 
- Make sure you respect clean code guidelines.
- Some parts of the already existing code are bad. Your job is to refactor them.
- Read the assignement carefully.
- Read the code thoroughly before you begin coding.

## Description
This mini project is a data app that revolves around resume text classification.

You are given a `dataset` that contains a number of resumes with their labels.

Each row of the dataset contains:
- Label 1, 2, ..., 13 You will find the resume labels map under data_ml_assignment/constants
- Resume text

The project contains by default:
- A baseline `naive bayes pipeline` trained on the aforementioned dataset
- An `API` that exposes an `inference endpoint` for predictions using the baseline pipeline
- A streamlit dashboard divided on three parts `(Exploratory Data Analysis, Training, Inference)`

## Assignment
### 1 - Code Refactoring
`Streamlit` is a component-based data app creator that allows you to create interactive dashboards using Python. 

While Streamlit is easy to use by "non frontenders", it can easily turn into a complicated piece of code.

As mentioned previously, the streamlit dashboard you have at hand is divided into 3 sections:
- Exploratory Data Analysis
- Training
- Inference

The code for the dashboard is written into one long Python (`dashboard.py`) script which makes it long, unoptimized, hard to read, hard to maintain, and hard to upgrade.

Your job is to:
- Rewrite the code while respecting `clean code` guidelines.
- `Refactor` the script and dissociate the components.
- Create the appropriate `abstraction` to make it easy to add components on top of the existing code.

`Bonus points`: if you pinpoint any other code anomalies across the whole project and correct them.

### 2 - Exploratory Data Analysis
In this section, you are asked to explore the dataset you are provided and derive insights from it:
- Statistical Descriptions
- Charts

Your EDA must be added to the first section of the streamlit dashboard.

P.S: You can add data processing in this section if needed.

![](./static/eda.png)

Hints: Please refer to the [documentation](https://docs.streamlit.io/library/api-reference) to learn more on how to use Streamlit `widgets` in order to display: `pandas dataframes`, `charts`, `tables`, etc, as well as interactive components: `text inputs`, `buttons`, `sliders`, etc.

### 3 - Training 
In this section, you are asked to `beat` the baseline pipeline. 

The trained pipeline is a combination of a Count Vectorizer and a Naive Bayes model

The goal is to capitalize on what you have discovered during the `EDA phase` and use the insights you derived in order to create a pipeline that performs `better` than the baseline you were provided.

The higher the `F1 score` the better.

You can `trigger` the baseline pipeline `training` in the `second` section of the `dashboard`.

Choose the `name` of the pipeline and whether you want to `serialize` it.

![](./static/training.png)

Click `Train pipeline` and wait until the training is done...

![](./static/training_current.png)

Once done, you will be able to see the F1 score as well as the confusion matrix.

P.S: If you chose the `save option` at the beginning of the training, you will be able to see the serialized pipeline under `models/pipeline_name`

![](./static/training_result.png)

Hints: 
- Make sure to change the training pipeline before you can trigger the training of your own from the dashboard.
- Make sure to add a vectorization method to your pipeline if missing.
- Your model must respect the abstraction used to build the baseline

### 4 - Inference

`Inference` is just a fancy word to say `prediction`.

In the third section of the dashboard, you can `choose` different `resumes` and run the serialized pipeline against them.

![](./static/inference.png)

The example shows an inference for a Java Developer resume:

![](./static/inference_done.png)

In this section, you are asked to: 
- Create an `endpoint` that allows you to `save` the prediction results into a `SQlite table`.
- Display the `contents` of the SQlite table after each inference run.

Hints: Think about using `SQLALchemy`

### 5 - Unit testing

As mentioned previously, your code should be unit tested. 

Hints: Use `pytest` for your unit tests as well as `mocks` for external services.

## Git Best Practices

- **Write Meaningful Commit Messages**: Each commit message should be clear and concise, describing the changes made. Use the format:
  ```
  <type>: <short description>
  ```
  Examples:  
  - `feat: add extraction task for ETL pipeline`  
  - `fix: resolve bug in transform job schema`  
  - `refactor: split ETL script into modular tasks`

- **Commit Small, Logical Changes**: Avoid bundling unrelated changes in one commit.

- **Review Before Committing**: Ensure clean and tested code before committing.
- **...

[This guide](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) provides detailed insights into writing better commit messages, branching strategies, and overall Git workflows.


```
ML-Internship-Home-Assignment
├─ api.log
├─ api_pid.txt
├─ dashboard.py
├─ data
│  ├─ processed
│  └─ raw
│     └─ resume.csv
├─ data_ml_assignment
│  ├─ api
│  │  ├─ constants.py
│  │  ├─ inference_route.py
│  │  ├─ main.py
│  │  ├─ schemas.py
│  │  ├─ server.py
│  │  ├─ __init__.py
│  │  └─ __pycache__
│  │     ├─ constants.cpython-311.pyc
│  │     ├─ constants.cpython-312.pyc
│  │     ├─ inference_route.cpython-311.pyc
│  │     ├─ inference_route.cpython-312.pyc
│  │     ├─ main.cpython-312.pyc
│  │     ├─ schemas.cpython-311.pyc
│  │     ├─ schemas.cpython-312.pyc
│  │     ├─ server.cpython-311.pyc
│  │     ├─ server.cpython-312.pyc
│  │     ├─ __init__.cpython-311.pyc
│  │     └─ __init__.cpython-312.pyc
│  ├─ constants.py
│  ├─ models
│  │  ├─ base_model.py
│  │  ├─ estimator_interface.py
│  │  ├─ naive_bayes_model.py
│  │  ├─ svc_model.py
│  │  ├─ xgbc_model.py
│  │  ├─ __init__.py
│  │  └─ __pycache__
│  │     ├─ base_model.cpython-311.pyc
│  │     ├─ base_model.cpython-312.pyc
│  │     ├─ estimator_interface.cpython-311.pyc
│  │     ├─ estimator_interface.cpython-312.pyc
│  │     ├─ naive_bayes_model.cpython-311.pyc
│  │     ├─ naive_bayes_model.cpython-312.pyc
│  │     ├─ __init__.cpython-311.pyc
│  │     └─ __init__.cpython-312.pyc
│  ├─ training
│  │  ├─ train_pipeline.py
│  │  ├─ __init__.py
│  │  └─ __pycache__
│  │     ├─ train_pipeline.cpython-311.pyc
│  │     ├─ train_pipeline.cpython-312.pyc
│  │     ├─ __init__.cpython-311.pyc
│  │     └─ __init__.cpython-312.pyc
│  ├─ utils
│  │  ├─ plot_utils.py
│  │  ├─ __init__.py
│  │  └─ __pycache__
│  │     ├─ plot_utils.cpython-311.pyc
│  │     ├─ plot_utils.cpython-312.pyc
│  │     ├─ __init__.cpython-311.pyc
│  │     └─ __init__.cpython-312.pyc
│  ├─ __init__.py
│  └─ __pycache__
│     ├─ constants.cpython-311.pyc
│     ├─ constants.cpython-312.pyc
│     ├─ __init__.cpython-311.pyc
│     └─ __init__.cpython-312.pyc
├─ models
│  ├─ .joblib
│  └─ naive_bayes_pipeline.joblib
├─ pyproject.toml
├─ README.md
├─ reports
│  └─ cm_plot.png
├─ run.sh
├─ samples
│  ├─ BUSINESS_ANALYST.txt
│  ├─ BUSINESS_INTELLIGENCE.txt
│  ├─ DOT_NET_DEVELOPER.txt
│  ├─ HELP_DESK_AND_SUPPORT.txt
│  ├─ INFORMATICA_DEVELOPER.txt
│  ├─ JAVA_DEVELOPER.txt
│  ├─ NETWORK_AND_SYSTEM_ADMINISTRATOR.txt
│  ├─ ORACLE_DBA.txt
│  ├─ PROJECT_MANAGER.txt
│  ├─ QUALITY_ASSURANCE.txt
│  ├─ SAP.txt
│  ├─ SHAREPOINT_DEVELOPER.txt
│  ├─ SQL_DEVELOPER.txt
│  └─ WEB_DEVELOPER.txt
├─ static
│  ├─ eda.png
│  ├─ inference.png
│  ├─ inference_done.png
│  ├─ training.png
│  ├─ training_current.png
│  └─ training_result.png
├─ streamlit.log
├─ streamlit_pid.txt
├─ tests
│  └─ __init__.py
└─ __init__.py

```