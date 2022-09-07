# base image
FROM python:3.9-slim

# set the working dir to /app
WORKDIR /app

# copy requirenents,txt
COPY requirements.txt /app

# change the dir and install libs from requirements.txt
RUN pip install --upgrade pip \
  && pip install -r requirements.txt

# to export the result to the outside of the container
# EXPOSE 8080

# create mountpoint
VOLUME /app

CMD ["python3", "main.py", "-I=demo2.csv", "-T=price", "-D=make aspiration body_style"]
