import torch
import torch.nn as nn
from models.modelv1 import ModelV1
from dataset import get_train, get_test, data_transforms
import data.data_creator as data_creator
import keyboard
import time
import warnings
from pynput.keyboard import Key, Controller

warnings.filterwarnings("ignore")

# Load model
model = ModelV1(in_shape=1, hidden=128, n_classes=7)
model.load_state_dict(torch.load('modelv1.pth', map_location='cpu'))
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
    for i in range(len(predicted_char)):
        keyb.press(predicted_char[i])
    keyb.press(Key.space)
    time.sleep(0.1)  # slight delay to simulate typing
    for i in range(len(predicted_char)):
        keyb.release(predicted_char[i])
    keyb.release(Key.space)

try:
    while True:
        if keyboard.is_pressed('esc'):
            print('\n[EXIT] ESC key pressed. Stopping loop.')
            break

        if keyboard.is_pressed('enter'):
            if not active:
                print('\n[INFO] Starting predictions...')
                active = True
                time.sleep(0.3)  # prevent key spam
            else:
                print('\n[INFO] Stopping predictions...')
                active = False
                time.sleep(0.3)

        if active:
            predict_and_plot(model)
            time.sleep(1)  # delay between inferences

        time.sleep(0.01)  # prevent CPU from melting
except KeyboardInterrupt:
    print('\n[EXIT] KeyboardInterrupt received.')
