import json
import logging
import openai
import os

# import openai_secret_manager

from slack_bolt import App
from slack_bolt.adapter.aws_lambda import SlackRequestHandler

# Set up OpenAI API credentials
# assert "openai" in openai_secret_manager.get_services()
# secrets = openai_secret_manager.get_secret("openai")
openai.api_key = os.environ["OPENAI_API_KEY"]

# Set up Slack API credentials
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]

# Initialize Slack app
app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)

SlackRequestHandler.clear_all_log_handlers()
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)


# Handle the app_mention event
@app.event("app_mention")
def handle_app_mentions(body, say, logger):
    print("handle_app_mention", body)

    event = body["event"]

    # Get the text of the incoming message
    message_text = event["text"]

    # Send the message to OpenAI and get the response
    response_stream = (
        openai.Completion.create(
            engine="davinci",
            prompt=message_text,
            max_tokens=60,
            n=1,
            stop=None,
            temperature=0.5,
        )
        .get("choices")[0]
        .get("text")
    )

    # Post the response back to the Slack channel
    channel_id = event["channel"]
    thread_ts = event["ts"]
    response = {"text": response_stream, "thread_ts": thread_ts}
    app.client.conversations_replies(channel=channel_id, **response)


# Define the message handler function
@app.event("message")
def handle_message(body, say, logger):
    print("handle_message", body)

    event = body["event"]

    # If the incoming message was generated by a bot or Slack app, then ignore it
    if "bot_id" in event or "app_id" in event:
        return

    # Get the text of the incoming message
    message_text = event["text"]

    # Send the message to OpenAI and get the response
    response_stream = (
        openai.Completion.create(
            engine="davinci",
            prompt=message_text,
            max_tokens=60,
            n=1,
            stop=None,
            temperature=0.5,
        )
        .get("choices")[0]
        .get("text")
    )

    # Post the response back to the Slack channel
    channel_id = event["channel"]
    thread_ts = event["ts"]
    response = {"text": response_stream, "thread_ts": thread_ts}
    app.client.conversations_replies(channel=channel_id, **response)


def lambda_handler(event, context):
    body = json.loads(event["body"])

    print("lambda_handler", body)

    if "challenge" in body:
        # Respond to the Slack Event Subscription Challenge
        return {
            "statusCode": 200,
            "headers": {"Content-type": "application/json"},
            "body": json.dumps({"challenge": body["challenge"]}),
        }

    # # Initialize the Slack app
    # app.start()

    # # Handle the message event
    # app.event("message")(handle_message)

    # # Handle the app_mention event
    # app.event("app_mention")(handle_app_mention)

    slack_handler = SlackRequestHandler(app=app)
    return slack_handler.handle(event, context)

    # # Return a success message
    # return {
    #     "statusCode": 200,
    #     "headers": {"Content-type": "application/json"},
    #     "body": json.dumps({"message": "Success"}),
    # }
