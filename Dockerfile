FROM tensorflow/tensorflow
WORKDIR ./
COPY . .
RUN pip install -r cat.txt
EXPOSE 8000
CMD ["gunicorn", "app:app"]
