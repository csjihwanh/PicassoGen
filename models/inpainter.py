from openai import OpenAI
import os 
import dotenv



response = client.images.generate(
    model="dall-e-3",
    prompt="clean background without any object in it.",
    size="1024x1024",
    quality="standard",
    n=1,
)

image_url = response.data[0].url
print(image_url)

class Inpainter:
    def __init__(self):
        dotenv.load_dotenv()

        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("The OPENAI_API_KEY environment variable is not set.")

        self.client = OpenAI()

    def in