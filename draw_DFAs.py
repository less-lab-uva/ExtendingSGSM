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
        for file_ending in ['svg', 'png']:
            for color in [True, False]:
                color_str = '_color' if color else ''
                prop.ltldfa.save_image(f'{args.save_folder}/{prop.name}_dfa{color_str}.{file_ending}', color=color)
                prop.reset_prop.ltldfa.save_image(f'{args.save_folder}/{prop.name}_dfa_R{color_str}.{file_ending}', color=color)



if __name__ == '__main__':
    main()
