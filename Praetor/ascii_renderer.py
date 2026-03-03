import os
from PIL import Image

ASCII_CHARS = ["@", "%", "#", "*", "+", "=", "-", ":", ".", " "]


def resize_image(image, new_width=80):
    width, height = image.size
    aspect_ratio = height / width
    new_height = int(aspect_ratio * new_width * 0.55)  # 0.55 corrects character aspect ratio
    resized_image = image.resize((new_width, new_height))
    return resized_image


def grayify(image):
    return image.convert("L")


def pixels_to_ascii(image):
    pixels = image.getdata()
    ascii_str = ""
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel // 25]  # 255/10 ~ 25
    return ascii_str


def image_to_ascii_art(image_path, output_width=80):
    if not os.path.exists(image_path):
        return "[Image Not Found]"

    try:
        image = Image.open(image_path)
    except Exception as e:
        return f"[Error loading image: {e}]"

    image = resize_image(image, output_width)
    image = grayify(image)

    ascii_str = pixels_to_ascii(image)
    img_width = image.width
    ascii_img = "\n".join([ascii_str[i:(i + img_width)] for i in range(0, len(ascii_str), img_width)])
    return ascii_img


def render_portrait(philosopher):
    portraits = {
        "socrates": "../Assets/portraits/Socrates.png",
        "plato": "../Assets/portraits/Plato.png",
        # Add more philosopher portrait paths here
    }

    path = portraits.get(philosopher.lower())
    if not path:
        return f"[No portrait available for {philosopher}]"
    return image_to_ascii_art(path)


# Test call for CLI display
if __name__ == "__main__":
    print(render_portrait("plato"))