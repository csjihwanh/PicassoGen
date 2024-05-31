import os 
import dotenv
from PIL import Image, ImageOps
import numpy as np
import cv2
import io

from openai import OpenAI

from configs import settings
from utils import download_image

class Inpainter:
    def __init__(self):
        dotenv.load_dotenv()

        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("The OPENAI_API_KEY environment variable is not set.")

        self.client = OpenAI()

        # config value setting 
        self.image_height = settings['image_height']
        self.image_width = settings['image_width']
        self.image_size = f'{self.image_width}x{self.image_height}'
        self.inpaint_model = settings['inpaint_model']
        self.bgr_path = settings['bgr_path']
        self.inpaint_path = settings['inpaint_path']

    def __call__(self, 
                 mask:dict
                 ):
        return self.inpaint(mask)

    def _inpaint_single(self, image_path, mask, prompt):
        """
        args: 
          image: path
          mask: io
          prompt: str

        return: img_url 
        """

        print(image_path, mask)

        response = self.client.images.edit(
            model=self.inpaint_model,
            image=open(image_path, "rb"),
            mask=open('assets/masks/masked.png', 'rb'),
            prompt=prompt,
            n=1,
            size=self.image_size 
        )

        return response.data[0].url

    def _draw_background(self):
        bgr_image = Image.new('RGB', (self.image_width, self.image_height), color='white')
        bgr_image.save(self.bgr_path)

    
    def _create_mask(self, mask:list) -> np.ndarray:
        """
        Creates a mask image based on center coordinates, width, and height.

        Args:
            center_x (int): X coordinate of the rectangle center.
            center_y (int): Y coordinate of the rectangle center.
            width (int): Width of the rectangle.
            height (int): Height of the rectangle.

        Returns:
            np.ndarray: Generated mask image.
        """
        center_x, center_y, width, height = mask
        mask = np.zeros((self.image_width, self.image_height), dtype="uint8")
        top_left_x = center_x - width // 2
        top_left_y = center_y - height // 2
        bottom_right_x = center_x + width // 2
        bottom_right_y = center_y + height // 2
        mask = cv2.rectangle(mask, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), 255, -1)

        return mask
    
    def _make_transparent_mask(self, base_image_path, mask):

        base_image = Image.open(base_image_path).convert('RGBA')
        mask_image = Image.fromarray(mask).convert('L')

        # Resize the mask image to match the size of the base image if necessary
        if base_image.size != mask_image.size:
            mask_image = mask_image.resize(base_image.size, Image.LANCZOS)

        # Invert the mask image (white areas will become black, black areas will become white)
        inverted_mask = ImageOps.invert(mask_image)

        # Create an alpha mask from the inverted mask
        alpha_mask = inverted_mask.point(lambda p: p > 128 and 255)

        # Apply the alpha mask to the base image
        base_image.putalpha(alpha_mask)

        # Save the result
        base_image.save('assets/masks/masked.png')


    def _mask_to_image_bytes(self,mask: np.ndarray) -> io.BytesIO:
        """
        Converts a mask (numpy array) to an in-memory image.

        Args:
            mask (np.ndarray): The mask to convert.

        Returns:
            io.BytesIO: The in-memory image.
        """
        # Convert the numpy array to a PIL image
        image = Image.fromarray(mask)
        
        # Save the image to a BytesIO object
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return img_byte_arr

    

    def inpaint(self, mask):
        object_names = mask['object_name']
        num_objects = mask['num_objects']
        position_list = mask['position_list']

        self._draw_background()
        prev_path = self.bgr_path

        for idx in range(num_objects):
            object_name = object_names[idx]
            position = position_list[idx]
            current_path = os.path.join(self.inpaint_path, f'inpaint_{idx}.png') 

            mask = self._create_mask(position)
            #mask = self._mask_to_image_bytes(mask)
            self._make_transparent_mask(prev_path, mask)

            prompt = f'inpaint {object_name} on the mask'
            print(prompt)

            img_url = self._inpaint_single(prev_path, mask, prompt)
            download_image(img_url, current_path)

            prev_path = current_path

