from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

response = client.chat.completions.create(
    model="llama3.1:8b",
    messages=[
        {"role":"user","content":"You are the lead architect for a production SaaS application with 10 million users. Design a scalable notification system supporting email, SMS, push notifications, and webhooks. The system must guarantee at-least-once delivery, support retries with exponential backoff, avoid duplicate notifications, tolerate regional outages, and process 100,000 events per second. Explain the database schema, API design, queue architecture, failure modes, observability, deployment strategy, and trade-offs between Kafka, RabbitMQ, and SQS."}
    ]
)

print(response.choices[0].message.content)