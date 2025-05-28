# Introduction

Hello everyone!  
Welcome to the **Brassflows Algorithm** — a project centered around a custom-built **Convolutional Neural Network (CNN)** designed to recognize key combinations from screenshots in real-time.

This neural network processes visual input and classifies it into one of several possible keypress combinations (such as `j`, `k`, `l`, or combinations like `jk`, `jl`, etc.). The model is trained on labeled screenshots that capture these combinations, enabling it to learn the visual features associated with each input.

The goal of this project is to demonstrate the ability to:
- Collect real-time data through screenshots.
- Train a CNN from scratch using PyTorch.
- Predict and output live keypresses based on screen content.

### Disclaimer

I’m fully aware that there are more efficient or conventional solutions to this type of problem — but that’s not the point of this project.

This approach was intentional. I'm currently focused on improving my skills in artificial intelligence and neural networks, particularly in understanding and building Convolutional Neural Networks (CNNs) from the ground up. By tackling a real, practical problem and applying deep learning as the core solution, I’m able to explore the full cycle: from data collection, preprocessing, and model architecture, to evaluation and real-time inference.

This project isn't just about getting the "right" answer — it's about learning how to engineer AI systems, getting hands-on experience with PyTorch, and pushing myself to develop solutions that bridge theory and application. I'm using this as a foundation for deeper work in AI, and a stepping stone toward building more complex, scalable, and efficient systems in the future.


## How this project came up

The story of how I got the idea for this project is quite interesting. As you can tell by the title of this, `brassflows-algorithm`, this algorithm was meant to be able to accomplish tasks present in this website called brassflows. It is meant for brass instrument players to practice fingerings for different notes on their instruments, however it is ranked. One of my friends came about, beat my rank, and said, I quote, "What are you going to do? Build an algorithm for it?" So that is exactly what I did! This model is **purely** for demo purposes and SHOULD NOT be used to cheat on this website. This serves as a way to show what a CNN is capable of and NOT a device meant for cheating or unfair playing.

# How this project was made

`This chapter is currently a work in progress, check again later to see if I completed it at some point.`

# How to Use

This project uses a trained Convolutional Neural Network (CNN) to recognize key combinations (`j`, `k`, `l`, or combinations of them) based on real-time screen captures. Here's how you can try it out for yourself:

---

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/brassflows-algorithm.git
cd brassflows-algorithm
```

## 2. Install Dependencies
Make sure you have Python 3.9+ installed. Then install the required packages:
```bash
pip install -r requirements.txt
```
---
## 3. Using the Model for Inference

Run infererence script:
`python runner.py`

- Wait till model is **initialized** (The cue for *Enter* key will display)
- Press the **Enter** key to start the process.
- Make sure you are on brassflows. The model will start taking screenshots of a segment of your display.
- Using those screenshots, it will predict what set of keys to be pushed. This will then translate to it pressing down keys for you.
- It will continue UNTIL you go into the `command line` or `terminal` and hit **Ctrl-C** killing the process.

---

# Notes

- The model currently uses static screenshots from a defined bounding box. For best results, ensure the visual format of the keys remains consistent.
- This project is intended for educational and experimental purposes. It’s a hands-on example of applying deep learning to real-time visual recognition tasks.

---

# Data

| Trial          | ModelV1.1 | ModelV1.5 | ModelV2  |
|----------------|-----------|-----------|----------|
| Train/Test 1   | 76.63%    | 97.98%    | 14.58%   |
| Train/Test 2   | 86.21%    | 95.76%    | 51.23%   |
| Train/Test 3   | 94.53%    | 100.0%    | 43.12%   |
| Latest         | ❌        | ✅        | ❌       |
