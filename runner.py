import torch
import torch.nn as nn
from models.modelv1 import ModelV1
from dataset import data_transforms
import data.data_creator as data_creator
import keyboard
import time
import warnings
from pynput.keyboard import Key, Controller
from uuid import uuid4
import random

warnings.filterwarnings("ignore")

# Load model
model = ModelV1(in_shape=1, hidden=128, n_classes=7)
model.load_state_dict(torch.load('saves/latest-stable_v010.pth', map_location='cpu'))
model.eval()

print(f'Running on torch version: {torch.__version__}')
char_to_idx = data_creator.classes
 
print('Press Enter to start. Press ESC to stop.')

active = False
keyb = Controller()

def predict_and_plot(model, device='cpu'):
    screenshot = data_creator.screenshot()
    screenshot = data_transforms(screenshot)
    screenshot = screenshot.unsqueeze(0)  # Add batch dim
    with torch.no_grad():
        output = model(screenshot)
    probs = output.softmax(dim=1).squeeze()
    _, predicted = torch.max(output, 1)
    predicted_label = predicted.item()
    predicted_char = list(char_to_idx.keys())[list(char_to_idx.values()).index(predicted_label)]
    print(f'\nOutput: {probs}')
    print(f'Predicted label: {predicted_label}, Character: {predicted_char}')
    for char in predicted_char:
        keyb.press(char)
    keyb.press(Key.space)
    time.sleep(random.random())
    for char in predicted_char: 
        keyb.release(char)
    keyb.release(Key.space)

# --- NEW: Handle keypress  for screenshot ---
def handle_keypress(e):
    global active
    if e.name == 'esc':
        print('\n[EXIT] ESC key pressed. Stopping loop.')
        exit()
    elif e.name == 'enter':
        active = not active
        print(f'\n[INFO] {"Starting" if active else "Stopping"} predictions...')
    elif e.name == '1': 
        print('\n[INFO] Taking screenshot...')
        screenshot = data_creator.screenshot()
        screenshot.save(f'errors/{uuid4()}.jpg')

# Register keypress callback
keyboard.on_press(handle_keypress)
try:
    while True:
        if active:
            predict_and_plot(model)
        time.sleep(0.01)  # don't burn CPU
except KeyboardInterrupt:
    print('\n[EXIT] KeyboardInterrupt received.')
