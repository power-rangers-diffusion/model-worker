import json

import boto3

sqs = boto3.resource("sqs")
queue = sqs.get_queue_by_name(QueueName="inference-request.fifo")

messages = queue.receive_messages(WaitTimeSeconds=20, MaxNumberOfMessages=1)
for message in messages:
    body = json.loads(message.body)

    message.delete()