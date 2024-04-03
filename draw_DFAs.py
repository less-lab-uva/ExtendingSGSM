import argparse
import os
from pathlib import Path
from properties import all_properties


def main():
    parser = argparse.ArgumentParser(prog='Property checker')
    parser.add_argument('-s', '--save_folder', type=Path, default='./dfa_images/')
    args = parser.parse_args()
    os.makedirs(args.save_folder, exist_ok=True)
    for prop in all_properties:
        img = prop.ltldfa.get_pydot_image(svg=True)
        image_file = f'{args.save_folder}/{prop.name}_dfa.svg'
        with open(image_file, 'wb') as f:
            f.write(img)


if __name__ == '__main__':
    main()
