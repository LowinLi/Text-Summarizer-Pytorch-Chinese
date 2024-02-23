FROM python:3.13.0a4

# Copy the current directory contents into the container at /app
COPY / .

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
RUN pip install -r requirements-web.txt -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com


# Define environment variable

ENV TZ Asia/Shanghai

# Run app.py when the container launche
CMD ["python", "web.py"]