import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import PIL

PIL.Image.MAX_IMAGE_PIXELS = 11881676800


class ImageRotatorApp:
    def __init__(self, root):
        self.root = root
        root.title("Image Rotator")

        self.angle = 0
        self.image = None
        self.photo_image = None
        self.image_path = None
        self.max_width, self.max_height = 750, 750  # Max dimensions

        self.canvas = tk.Canvas(root, width=self.max_width, height=self.max_height)
        self.canvas.pack()

        btn_load = tk.Button(root, text="Load Image", command=self.load_image)
        btn_load.pack()

        btn_save = tk.Button(root, text="Save Rotation Angle", command=self.save_rotation_angle)
        btn_save.pack()

        self.label_angle = tk.Label(root, text=f"Rotation: {self.angle}°")
        self.label_angle.pack()

        # Bind arrow keys to rotation functions
        root.bind('<Left>', lambda event: self.rotate_image(-1))
        root.bind('<Right>', lambda event: self.rotate_image(1))

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_path = file_path
            image = Image.open(file_path)
            self.image = self.resize_image(image)
            if os.path.exists(os.path.join('rotations/',
                                           (os.path.basename(self.image_path).split('.')[0]).split("_")[0] + '.txt')):
                with open(os.path.join('rotations/',
                                       (os.path.basename(self.image_path).split('.')[0]).split("_")[0] + '.txt'),
                          'r') as file:
                    self.angle = int(file.read())
            else:
                self.angle = 0  # Reset the angle when a new image is loaded
            self.label_angle.config(text=f"Rotation: {self.angle}°")
            self.update_canvas()

    def save_rotation_angle(self):
        if self.image_path and self.image:
            rotation_directory = './rotations/'
            os.makedirs(rotation_directory, exist_ok=True)
            filename = (os.path.basename(self.image_path).split('.')[0]).split("_")[0] + '.txt'
            with open(os.path.join(rotation_directory, filename), 'w') as file:
                file.write(f'{self.angle}')
            print(f"Rotation angle saved to {os.path.join(rotation_directory, filename)}")

    def resize_image(self, image):
        # Resize the image to fit within the max dimensions while maintaining aspect ratio
        ratio = min(self.max_width / image.width, self.max_height / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        return image.resize(new_size, Image.BILINEAR)

    def rotate_image(self, angle):
        if self.image:
            self.angle = (self.angle + angle) % 360
            rotated_image = self.image.rotate(-self.angle, expand=True)
            self.update_canvas(image=rotated_image)
            self.label_angle.config(text=f"Rotation: {self.angle}°")

    def update_canvas(self, image=None):
        if image is None:
            image = self.image

        self.photo_image = ImageTk.PhotoImage(image)
        self.canvas.config(width=image.width, height=image.height)
        self.canvas.create_image(image.width // 2, image.height // 2, image=self.photo_image, anchor=tk.CENTER)


def main():
    root = tk.Tk()
    app = ImageRotatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
