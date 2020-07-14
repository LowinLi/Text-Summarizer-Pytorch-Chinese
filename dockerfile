FROM python:3.6

# Copy the current directory contents into the container at /app
COPY /web/ .
COPY /requirements-web.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
RUN pip install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com


# Define environment variable

ENV TZ Asia/Shanghai

# Run app.py when the container launche
CMD ["python", "web.py"]