# Cloud Services for Data Science

**Concept**  
Cloud platforms (AWS, GCP, Azure) offer scalable storage and compute. You can automate tasks, access data, and train models at scale. For this module, boto3 is optional.

### Example
```python
# List S3 buckets using boto3 (requires AWS credentials and boto3)
try:
    import boto3
    s3 = boto3.client("s3")
    buckets = s3.list_buckets()
    print("Buckets:", [b["Name"] for b in buckets["Buckets"]])
except ImportError:
    print("boto3 not installed. Run: pip install boto3")
except Exception as e:
    print("Cloud error:", e)
```

### Exercise
"""
List all S3 bucket names (use boto3 and your AWS credentials).
"""
```python
# Your code here
```

### Quiz
**Q1:** Which AWS service is used for object storage?
- A) EC2
- B) Lambda
- C) S3
- D) SageMaker
**A:** C

**Q2:** What Python library is used for AWS automation?
**A:** boto3