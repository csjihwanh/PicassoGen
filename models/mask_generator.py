import autogen
from typing import List, Dict, Annotated
import os
import json
import logging
import dotenv
import cv2
import numpy as np
from configs import settings

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def mask_generation_tool(
        object_name: Annotated[List[str], "Names of each object with personality"],
        num_objects: Annotated[int, "Number of objects"],
        position_list: Annotated[List[List[int]], "Position of each rectangle mask: [center_x, center_y, width, height]"],
        output_dir: str = "assets/masks"
) -> Dict[str, any]:
    """
    Generates a dictionary containing object names, number of objects, and positions of rectangle masks.

    Args:
        object_name (List[str]): Name of each object with personality.
        num_objects (int): Number of rectangle masks.
        position_list (List[List[int]]): List of positions and sizes for each rectangle mask.
        output_dir (str): Directory where the masks will be saved (not used here).

    Returns:
        Dict[str, any]: Dictionary containing object names, number of objects, and positions of masks.
    """

    # Validate input
    if len(position_list) != num_objects:
        error_msg = "The length of position_list does not match num_objects."
        logging.error(error_msg)
        return {"error": error_msg}

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory at {output_dir}")

    masks = []
    for i in range(num_objects):
        try:
            center_x, center_y, width, height = position_list[i]
            mask = (center_x, center_y, width, height)
            masks.append({
                "object_name": object_name[i],
                "mask": mask
            })
        except Exception as e:
            error_msg = f"Error generating mask for {object_name[i]}_{i}: {str(e)}"
            logging.error(error_msg)
            return {"error": error_msg}

    result = {
        "object_name": object_name,
        "num_objects": num_objects,
        "position_list": position_list
    }

    json_output_path = os.path.join(output_dir, 'masks_data.json')
    try:
        with open(json_output_path, 'w') as json_file:
            json.dump(result, json_file)
        logging.info(f"Data saved to {json_output_path}")
    except Exception as e:
        error_msg = f"Error saving data to JSON file: {str(e)}"
        logging.error(error_msg)
        return error_msg

    return "Great! All data is saved to masks_data.json. TERMINATE"


class MaskGenerator:
    def __init__(self):
        
        self.image_height = settings['image_height']
        self.image_width = settings['image_width']
        self.image_size = f'{self.image_width}x{self.image_height}'
        self.model_name = settings['mask_gen_model']

        # LLM configuration

        config_list = autogen.config_list_from_json(
            env_or_file="configs/OAI_CONFIG_LIST_4.json",
            filter_dict={"model": [self.model_name]}
        )

        seed = 1
        llm_config = {
            "cache_seed": seed,
            "temperature": 0,
            "config_list": config_list,
            "timeout": 120,
        }
        self.llm_config = llm_config

        self.position_bot = autogen.AssistantAgent(
            name="position_bot",
            system_message=(
                "You are responsible for determining the positions and sizes of rectangles. Ensure the rectangles do not overlap and are within the image boundaries. "
                f"The image size is {self.image_size}. You must provide the position_list (List[List[int]]) to the position_verifier_bot. "
                "Each list item should be formatted as [center_x, center_y, width, height]. Ensure that the bounding boxes do not overlap or go beyond the image boundaries. "
                "If needed, make reasonable guesses to position the objects naturally within the scene described by the user prompt."
            ),
            llm_config=self.llm_config
        )

        self.position_verifier_bot = autogen.AssistantAgent(
            name="position_verifier_bot",
            system_message=(
                "You are responsible for verifying the naturalness and correctness of the positions and sizes of the rectangles based on the objects' names and the given positions and sizes. "
                f"The images are of size {self.image_size}. Each bounding box should be in the format of (object name [center_x, center_y, width, height]). "
                "If the positions and sizes do not match the given prompt naturally, ask the position bot to set the positions again, providing a concrete reason. "
                "If the positions are good, provide the position_list (List[List[int]]) to the mask_generation_bot. Otherwise, request repositioning from the position_bot. "
                f"Ensure the positions and sizes do not exceed the boundaries of a {self.image_size} image."
            ),
            llm_config=self.llm_config
        )

        self.mask_generation_bot = autogen.AssistantAgent(
            name="mask_generation_bot",
            system_message=(
                "You are responsible for generating rectangle masks based on given positions and sizes. "
                "Use the provided functions to create and return masks as dictionary. The dictionary has \"object_name\", \"num_objects\", \"position_list\", \"masks\" "
                "If the object name, position, and size do not align naturally, ask the position bot to reassign the position, providing a detailed reason. "
                f"The images are of size {self.image_size}. Each bounding box should be in the format of (object name [center_x, center_y, width, height]). "
                "Ensure that the bounding boxes do not overlap or go beyond the image boundaries. "
                "After finishing all the task, save it in json format file using the given method."
                "Reply TERMINATE when the task is complete."
            ),
            llm_config=self.llm_config
        )

        self.user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            human_input_mode="NEVER",
            max_consecutive_auto_reply=5,
            code_execution_config=False
        )

        self.group_chat = autogen.GroupChat(
            agents=[self.user_proxy, self.position_bot, self.mask_generation_bot, self.position_verifier_bot],
            messages=[],
            max_round=20
        )
        self.manager = autogen.GroupChatManager(groupchat=self.group_chat, llm_config=self.llm_config)

        self._register_tools()

    def _register_tools(self):
        self.user_proxy.register_for_execution(self.initiate_chat)

        self.mask_generation_bot.register_for_llm(name="mask_generator",description="mask generation tool")(mask_generation_tool)
        self.user_proxy.register_for_execution(name="mask_generator")(mask_generation_tool)

    def initiate_chat(self, first_message: str) -> None:
        self.user_proxy.initiate_chat(
            self.manager,
            message=first_message
        )


# Example usage
if __name__ == "__main__":
    

    generator = MaskGenerator()
    prompt = "Draw three balls "
    
    result = generator.initiate_chat(prompt)


    print(result)

    print(type(result))
