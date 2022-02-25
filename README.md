## Wafer Fault Detection

----
## Link

<a href="https://wafer-fault--detection.herokuapp.com/">Heroku App Link</a>


#### Problem Statement:

  The goal is to build a program to predict the faulty wafers with the information previously gathered by using classification methods in Machine Learning.

#### Architecture

<a href="https://ibb.co/rFNGBFh"><img src="https://i.ibb.co/PF0Z2Fy/Wafer-Fault-Detection-Architecture.png" alt="Wafer-Fault-Detection-Architecture" border="0"></a>

#### Data Description

  - As per the Data Sharing agreement, we will receive data files in a shared location by the client, in the pre-decided formats. The format of files agreed is comma-separated values(.csv) files, with 590 columns each representing a sensor in a wafer, wafer name column, and an output column with values +1 (Working wafer) and -1(Faulty wafer).
  - As per the data sharing agreement, we will also get two schema files from the client, which contains all the relevant information about the training and prediction datafiles.


#### Data Validation & Transformation

  Once we gather all the data from the client. We will start the data validation process as the data sharing agreement and the requirements of machine learning process as a part of our training process.
  1. **Filename validation**: First we will start with the file name validation, as per the schema file given for the training datasets. We will create a regex pattern as per the name given in the schema file to use for validation. After validating the pattern in the name, we will check for the length of date and time in the file name. If all the values are as per the schema file, we will move such files to Good_Data_Folder. Else, we will move such files to Bad_Data_Folder.
  2. No. of Columns: We will validate the no. of columns present in each file in the Good_Data_Folder, if the no. of columns present in a file matches the no. of columns present in the schema file for training, that file will be retained in the same folder. Else, that file will be moved to Bad_Data_Folder.
  3. Name of Columns: The names of the columns present in each file in the Good_Data_Folder is validated and should be as per the schema file. Else, those files will be moved to the Bad_Data_Folder.
  4. Datatype of Columns: The datatypes of the columns present in each file in the Good_Data_Folder is validated and should be as per the schema file. Else, those files will be moved to the Bad_Data_Folder.
  5. Null values in columns: If any of the columns in a file have all the values as Null or missing, we discard such a file and move it to Bad_Data_Folder. And, we will replace the other null values with a string code “Null”.


#### Data Insertion into Database

  - Database Creation: Create a SQLite database, if it is not already present in the given directory. If it is present, open the database by connecting to that database.
  - Table Creation in the Database: Create a table in the database with name “Good_Data” for inserting the files in the Good_Data_Folder based on given column names and datatypes in the schema file, if it is not already present in the database. If that table is already present in the database, no need to create a new table.
  - Insertion of files in the table: All the files in the Good_Data_Folder are inserted in this table. If any files are raising errors while inserting the data to the table due to the invalid datatypes, those files will be moved to the Bad_Data_Folder.

#### Export the data to a csv file

  - The data from the database will be exported to the csv file, and is used for model training in the later stages.
  - All the files in the Bad_Data_Folder will be moved to Archives, as we want to show the rejected files to the client.
  - All the files in the Good_Data_Folder will be deleted as we have already captured this data in our database.

#### Data Pre-processing
  We will read the exported csv file, and impute all the values with Null String code using KNN Imputer. We will also perform necessary feature engineering techniques for as part of our pre-processing step, like dropping all the columns with zero standard deviation etc.

#### Data Clustering

  We are using an semi-supervised machine learning process. So, once our data is clean, we cluster the data into different clusters (using KMeans Clustering Algorithm) which we later use for training different models on each cluster, this will increase the overall accuracy of the project. So, first we divide the data based on implicit patterns in the data. Then, we give a no. to each cluster, and add new column which consists of cluster name.

#### Model Training

  By using four machine learning algorithms, “Support-Vector Classifier”, “Decision-Tree Classifier”, “Random-Forest Classifier”, and “XGBoost Classifier”, we will train each cluster by Grid Searching few hyperparameters that are already defined using Cross-validation. We decide the best model for each cluster based on their auc scores, if auc score cannot be produced due to the testing dataset, we will use accuracy score. Once we find the best models for each cluster, we will save them in the models folder with their clusters numbers in their names and folder names.


#### Deployment

  - After training the model in the local system, and testing it. We will create a CICD pipeline using circleci and dockerhub for our model deployment.
  - We will deploy trained model in Heroku platform.
  - As the training takes a lot of time, we disabled the option of training the model using the website.
  - The model training can be performed using postman, by providing the location of the training batch files {"folderPath":"Training_Batch_files"}.


#### Prediction

  - Before predicting the output of the data present in the data files, we have to perform some similar actions we did in the training process. These steps are required to insert our data  for prediction. We will also have a schema file for prediction, with a little difference when compared to the schema file for training i.e. no output column present in the prediction schema file or in the batch files given for the prediction. We will perform the following steps:
    - File name validation
    - File type validation
    - No. of columns validation
    - Name of columns validation
    - Datatypes of the columns validation
    - Null values validation and transformation
  - The validated data will inserted into a database, and after completing the insertion process for all the files. The files present in the  Good_Data_Folder will be deleted and Bad_Data_Folder will be moved to archive.
  - The data from the database will be exported into a csv file and similar pre-processing steps will be form on the data loaded from this csv file.
  - After the pre-processing steps, the data will be divided into the clusters by using the already trained Kmeans clustering model.
  And, the best model created for each cluster will be used for prediction and all the data will be compiled together with their respective wafer names.
  - Finally, the prediction results will be exported as a csv file.

----

## Create a file "Dockerfile" with below content

```
FROM python:3.7
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT [ "python" ]
CMD [ "main.py" ]
```

## Create a "Procfile" with following content
```
web: gunicorn main:app
```

## create a file ".circleci\config.yml" with following content
```
version: 2.1
orbs:
  heroku: circleci/heroku@1.0.1
jobs:
  build-and-test:
    executor: heroku/default
    docker:
      - image: circleci/python:3.6.2-stretch-browsers
        auth:
          username: mydockerhub-user
          password: $DOCKERHUB_PASSWORD  # context / project UI env-var reference
    steps:
      - checkout
      - restore_cache:
          key: deps1-{{ .Branch }}-{{ checksum "requirements.txt" }}
      - run:
          name: Install Python deps in a venv
          command: |
            echo 'export TAG=0.1.${CIRCLE_BUILD_NUM}' >> $BASH_ENV
            echo 'export IMAGE_NAME=python-circleci-docker' >> $BASH_ENV
            python3 -m venv venv
            . venv/bin/activate
            pip install --upgrade pip
            pip install -r requirements.txt
      - save_cache:
          key: deps1-{{ .Branch }}-{{ checksum "requirements.txt" }}
          paths:
            - "venv"
      - run:
          command: |
            . venv/bin/activate
            python -m pytest -v tests/test_script.py
      - store_artifacts:
          path: test-reports/
          destination: tr1
      - store_test_results:
          path: test-reports/
      - setup_remote_docker:
          version: 19.03.13
      - run:
          name: Build and push Docker image
          command: |
            docker build -t $DOCKERHUB_USER/$IMAGE_NAME:$TAG .
            docker login -u $DOCKERHUB_USER -p $DOCKER_HUB_PASSWORD_USER docker.io
            docker push $DOCKERHUB_USER/$IMAGE_NAME:$TAG
  deploy:
    executor: heroku/default
    steps:
      - checkout
      - run:
          name: Storing previous commit
          command: |
            git rev-parse HEAD > ./commit.txt
      - heroku/install
      - setup_remote_docker:
          version: 18.06.0-ce
      - run:
          name: Pushing to heroku registry
          command: |
            heroku container:login
            #heroku ps:scale web=1 -a $HEROKU_APP_NAME
            heroku container:push web -a $HEROKU_APP_NAME
            heroku container:release web -a $HEROKU_APP_NAME

workflows:
  build-test-deploy:
    jobs:
      - build-and-test
      - deploy:
          requires:
            - build-and-test
          filters:
            branches:
              only:
                - main
```
## to create requirements.txt

```buildoutcfg
pip freeze>requirements.txt
```

## initialize git repo

```
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin
git push -u origin main
```

## create a account at circle ci

<a href="https://circleci.com/login/">Circle CI</a>

## setup your project

<a href="https://app.circleci.com/pipelines/github/saisrinivas-samoju/waferfaultdetection"> Setup project </a>

## Select project setting in CircleCI and below environment variable

>DOCKERHUB_USER
>DOCKER_HUB_PASSWORD_USER
>HEROKU_API_KEY
>HEROKU_APP_NAME
>HEROKU_EMAIL_ADDRESS

>DOCKER_IMAGE_NAME=<wafercircle258>

## to update the modification

```
git add .
git commit -m "proper message"
git push
```


<!-- ## #docker login -u $DOCKERHUB_USER -p $DOCKER_HUB_PASSWORD_USER docker.io -->
