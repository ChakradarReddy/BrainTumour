FROM tensorflow/tensorflow
WORKDIR ./
COPY . .
RUN pip install -r cat.txt
EXPOSE 8000
CMD ["gunicorn"  , "-b", "0.0.0.0:8000", "app:app"]

