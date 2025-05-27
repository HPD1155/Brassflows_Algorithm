import PIL.ImageGrab
import uuid
from os import makedirs, path
import keyboard  # new: this one does real-time key checking
import time

# Screenshot bounding box: (674, 363) (1272, 623)
# Key/Class table
classes = {
    '': 0,
    'j': 1,
    'jk': 2,
    'jl': 3,
    'kl': 4,
    'jkl': 5,
    'k': 6
}

def set_to_string(i_set):
    return ''.join(sorted(i_set))



def create_data_folders(table):
    for i in table.values():
        dir_path = f'data/train/{i}'
        if not path.exists(dir_path):
            makedirs(dir_path)

def get_current_keys():
    keys = set()
    for k in ['j', 'k', 'l']:
        if keyboard.is_pressed(k):
            keys.add(k)
    return keys

def main():
    create_data_folders(classes)

    while True:
        if keyboard.is_pressed('esc'):
            print("Exiting...")
            break

        if keyboard.is_pressed('space'):
            current_keys = get_current_keys()
            current_class = classes.get(set_to_string(current_keys), None)

            if current_class is not None:
                print(f"Capturing class: {current_class}, keys: {current_keys}")
                image = PIL.ImageGrab.grab(bbox=(674, 300, 1272, 700))
                image.save(f'data/train/{current_class}/{str(uuid.uuid4())}.jpg')

                time.sleep(0.3)  # debounce so it doesn’t spam
            else:
                print(f"Unknown key combo: {current_keys}")
                time.sleep(0.3)

if __name__ == "__main__":
    main()
