import os
import time
from PIL import Image

# === ASCII Palette ===
ASCII_CHARS = "@%#*+=-:. "

# === Philosopher ASCII Map ===
PHILOSOPHER_ASCII_ART = {
    "plato": "images/ascii_plato.png",
    "nietzsche": "images/ascii_nietzsche.png",
    "seneca": "images/ascii_seneca.png",
    "kierkegaard": "images/ascii_kierkegaard.png",
    "jung": "images/ascii_jung.png",
    "camus": "images/ascii_camus.png",
    "freud": "images/ascii_freud.png",
    "machiavelli": "images/ascii_machiavelli.png",
    "descartes": "images/ascii_descartes.png"
}

def image_to_ascii(image_path, width=80):
    try:
        img = Image.open(image_path).convert("L")  # Grayscale
        w, h = img.size
        aspect_ratio = h / w
        new_height = int(width * aspect_ratio * 0.55)
        img = img.resize((width, new_height))

        pixels = img.getdata()
        ascii_str = "".join(ASCII_CHARS[pixel * (len(ASCII_CHARS) - 1) // 255] for pixel in pixels)
        ascii_lines = [ascii_str[index:index+width] for index in range(0, len(ascii_str), width)]
        return ascii_lines
    except Exception as e:
        return ["[Error rendering image]"]

def render_ascii_art(philosopher):
    name = philosopher.lower()
    image_path = PHILOSOPHER_ASCII_ART.get(name)

    if image_path and os.path.exists(image_path):
        ascii_lines = image_to_ascii(image_path)
        print("\n")
        for line in ascii_lines:
            print(line)
            time.sleep(0.02)
        print("\n")
    else:
        print(f"[No ASCII image found for {philosopher}]\n")

# === Example Use ===
if __name__ == "__main__":
    chosen_model = "plato"  # dynamically detect in CLI later
    render_ascii_art(chosen_model)
