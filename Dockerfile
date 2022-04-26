FROM tensorflow/tensorflow
WORKDIR ./
COPY . .
RUN pip install -r cat.txt
EXPOSE 8080
CMD ["gunicorn", "app:app"]
