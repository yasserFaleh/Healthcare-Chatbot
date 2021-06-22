FROM python:3
RUN pip install pandas sklearn numpy nltk
WORKDIR /usr/src/app
COPY . .
CMD ["jv.py"]
ENTRYPOINT ["python3"]