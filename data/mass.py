import os
import random
import uuid
import shutil

def duplicate_random_files(x_folder, duplications_per_folder=100):
    for root, dirs, _ in os.walk(x_folder):
        for d in dirs:
            folder_path = os.path.join(root, d)
            all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            
            if not all_files:
                print(f"Skipping empty folder: {folder_path}")
                continue

            print(f"Duplicating {duplications_per_folder} files in: {folder_path}")

            for _ in range(duplications_per_folder):
                file_name = random.choice(all_files)
                src_path = os.path.join(folder_path, file_name)
                file_ext = os.path.splitext(file_name)[-1]
                new_name = f"{uuid.uuid4()}{file_ext}"
                dest_path = os.path.join(folder_path, new_name)
                shutil.copy2(src_path, dest_path)

def create_test():
    os.makedirs("data/test", exist_ok=True)
    for i in range(7):
        os.makedirs(f"data/test/{i}", exist_ok=True)
    
    for i in range(50):
        for j in range(7):
            class_folder = f"data/{j}"
            if os.path.exists(class_folder):
                all_files = [f for f in os.listdir(class_folder) if os.path.isfile(os.path.join(class_folder, f))]
                if all_files:
                    file_name = random.choice(all_files)
                    src_path = os.path.join(class_folder, file_name)
                    dest_path = os.path.join("data/test", str(j), f"{uuid.uuid4()}.jpg")
                    shutil.copy2(src_path, dest_path)

# Clone 50 of a single image
def clone_single_image(image_path, target_folder, num_clones=50):
    os.makedirs(target_folder, exist_ok=True)
    for _ in range(num_clones):
        new_name = f"{uuid.uuid4()}.jpg"
        dest_path = os.path.join(target_folder, new_name)
        shutil.copy2(image_path, dest_path)

if __name__ == '__main__':
    clone_single_image('errors/03b8996e-66b6-47f4-8f12-4e67debd2a61.jpg', 'data/train/1', num_clones=50)
    clone_single_image('errors/03b8996e-66b6-47f4-8f12-4e67debd2a61.jpg', 'data/test/1', num_clones=20)