from celery import Celery
from celery.result import AsyncResult
from flask import (
    Flask,
    make_response,
    render_template,
    request,
)


celery_app = Celery('tasks', backend='redis://redis', broker='redis://redis')
app = Flask(__name__)


@app.route('/sentimental', methods=['GET', 'POST'])
def sentimental_handler():
    if request.method == 'POST':
        target_msg = request.form['target_msg']
        sample = [target_msg]
        task = celery_app.send_task('tasks.predict', sample)
        return render_template('check.html', value=task.id)


@app.route('/sentimental_analysis/<task_id>', methods=['GET', 'POST'])
def sentimental_analysis_check_handler(task_id):
    task = AsyncResult(task_id, app=celery_app)
    if task.ready():
        if task.result >= 0.75:
           processed_text = "Your comment is positive."
        elif task.result >= 0.45:
           processed_text = "Your comment is neutral."
        else:
           processed_text = "Your comment is negative."
        
        text = f'Sentimental analysis: \n {processed_text}(Score: {str(task.result)[:5]})'
    else:
        text = 'IN_PROGRESS'

    response = make_response(text, 200)
    response.mimetype = "text/plain"
    return response


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
