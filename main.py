import argparse
import json

from models import MaskGenerator, Inpainter

def main() :
    parser = argparse.ArgumentParser(description="parser setting")

    parser.add_argument('--prompt', required=True, type=str, help='prompt')

    args = parser.parse_args()
    prompt = args.prompt

    mask_gen = MaskGenerator()
    inpainter = Inpainter()

    mask_gen.initiate_chat(prompt)
    mask_dict_path = 'assets/masks/masks_data.json'

    with open(mask_dict_path, 'r') as f:
        data = json.load(f)

    inpainter(data)

if __name__ == '__main__':
    main()
