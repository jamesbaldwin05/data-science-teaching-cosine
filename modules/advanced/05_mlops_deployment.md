# MLOps & Deployment

**Concept**  
MLOps is the practice of deploying, monitoring, and maintaining ML models in production. Key concepts: packaging (e.g., Docker), CI/CD pipelines, and monitoring performance.

### Example
```python
# Dockerfile example: containerize a Python app (not runnable here)
dockerfile = '''
FROM python:3.9
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
'''
print(dockerfile)
```

### Exercise
"""
Write a bash command to build a Docker image from the above Dockerfile.
"""
```python
# Your code here (print the docker build command as a string)
```

### Quiz
**Q1:** What does CI/CD stand for?
- A) Continuous Integration/Continuous Deployment
- B) Code Inspection/Code Debugging
- C) Compute Instance/Cloud Database
- D) Cloud Integration/Cloud Delivery
**A:** A

**Q2:** Which tool is commonly used to containerize Python apps?
**A:** docker