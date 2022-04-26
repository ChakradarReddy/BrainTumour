FROM python
WORKDIR ./
COPY . .
RUN pip install -r cat.txt
EXPOSE 3000
CMD ["gunicorn", "app:app"]
