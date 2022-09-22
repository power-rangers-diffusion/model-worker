import boto3

sqs = boto3.resource("sqs")

queue = sqs.get_queue_by_name(QueueName="inference-request.fifo")
messages = queue.receive_messages()
for m in messages:
    print(m.body)
    m.delete()