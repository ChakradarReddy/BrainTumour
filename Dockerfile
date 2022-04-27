FROM tensorflow/tensorflow
WORKDIR ./
COPY . .
RUN pip install -r cat.txt
EXPOSE 8000
ENV FLASK_APP =app.py
ENV FLASK_ENV = development
CMD ["flask"  , "run"]

