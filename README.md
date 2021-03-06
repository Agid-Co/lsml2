# LSML2 final project
## Sentimental analysis with BERT, Flask, Celery and Docker.

### Project purpose :
Create web application to judge positive, negative or neutral comment from inputted comments. 


### Dataset:
This dataset of Amazon Fine Foods reviews consists of 568,454 fine food reviews from amazon. The data span is from Oct 1999 to Oct 2012.

https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html


### Model:
This application model is fine-tuning BERT which was pre-trained BERT model that was trained on a huge dataset, as a starting point. 
I trained the model futher on our with above Amazon FIne Foods reviews dataset. 
The based model is from below huggingface bert-base-uncased model.

https://huggingface.co/bert-base-uncased


### Docker containers:
* Flask: web server 
* Client: Simple HTML with Javascript
* Celery: Work asynchronously 
* Redis: Cache and streaming engine
* MLFlow: Machine learning lifecycle
* DB: Postgre Database server


### How to train :
* pip install -r requirements.txt
* python train.py

The training process is below:

https://github.com/Agid-Co/lsml2/blob/master/lsml2_train.ipynb


### Running model:
* docker-compose up --build


### Checking the application:
* Web Interface - `http://localhost:9091/`

First screen:
Input your sentence for sentimental analysis

<p align='center'>
  <a href="https://github.com/Agid-Co/lsml2/blob/master/image/Web_Interface_1.PNG"><img src='https://github.com/Agid-Co/lsml2/blob/master/image/Web_Interface_1.PNG' width="600" height="250"></a>
</p>

Result:

You get positive, negative or neutral with score.

<p align='center'>
  <a href="https://github.com/Agid-Co/lsml2/blob/master/image/Web_Interface_2.PNG"><img src='https://github.com/Agid-Co/lsml2/blob/master/image/Web_Interface_2.PNG' width="600" height="100"></a>
</p>


* MLFLow web interface - `http://127.0.0.1:5000/#/`
