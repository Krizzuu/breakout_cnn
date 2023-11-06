FROM pytorch/pytorch


# Install any needed packages specified in requirements.txt
COPY requirements.txt /app/

# Set the working directory
WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code
COPY . /app

ENTRYPOINT ["python",  "-u", "main.py"]