# written by https://github.com/samuel1212703
import PIL.Image
import numpy as np
import cv2
import os

class AsciiFormatter():
    def __init__(self, image, ascii_scaling, pixel_mod, char_or_line, font_scale) -> None:
        self.image = image
        self.ASCII_SCALE = ascii_scaling
        self.PIXEL_MOD = pixel_mod
        self.CoL = char_or_line
        self.FONT_SCALE = font_scale

    def cls(self):
        clr_command = 'cls' if os.name == 'nt' else 'clear'
        os.system(clr_command)

    def image_to_ascii(self, w, h):
        aspect_ratio = w / h
        new_height = int(h * self.ASCII_SCALE)
        new_width = int(aspect_ratio * new_height)
        img = self.image.resize((new_width, new_height))
        img = img.convert('L')
        chars = ["@", "#", "S", "%", "?", "*", "+", ";", ":", ",", "."]
        pixels = img.getdata()
        new_pixels = [chars[pixel // self.PIXEL_MOD] for pixel in pixels]
        new_pixels = ''.join(new_pixels)
        ascii_image = [new_pixels[index:index + new_width] for index in range(0, len(new_pixels), new_width)]
        ascii_image = "\n".join(ascii_image)
        return ascii_image

    def ascii_to_image(self, ascii, w, h):
        img = np.zeros((h, w, 3), np.uint8)
        ascii_lines = ascii.split('\n')
        line_height = h / len(ascii_lines)
        char_width = w / len(ascii_lines[0])
        if self.CoL:
            for yi in range(0, len(ascii_lines)):
                for xi, char in enumerate(ascii_lines[yi]):
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    y = int(line_height / 2 + yi * line_height)
                    x = int(char_width / 2 + xi * char_width)
                    font_scale = line_height * self.FONT_SCALE
                    cv2.putText(img, char, (x, y), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            for i, line in enumerate(ascii_lines):
                font = cv2.FONT_HERSHEY_SIMPLEX
                y = int(line_height / 2 + i * line_height)
                font_scale = line_height * self.FONT_SCALE
                cv2.putText(img, line, (0, y), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        return img

class AsciiVideoFormatter():
    def __init__(self, video, fourcc='mp4v', font_scale=0.15, ascii_res=0.2, output_multiplier=1) -> None:
        self.cap = cv2.VideoCapture(video)
        self.FPS = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * output_multiplier)
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * output_multiplier)
        fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self.out = cv2.VideoWriter('output.mp4', fourcc, self.FPS, (self.frame_width, self.frame_height))
        self.APP_NAME = 'ASCII VIDEO FORMATTER'

        self.FONT_SCALE = font_scale * ascii_res
        self.ASCII_RESOLUTION_PERCENTAGE = ascii_res
        self.BY_CHAR_OR_LINE = True
        self.PRINT_IN_CONSOLE = not self.BY_CHAR_OR_LINE
        self.SHOW_ORIGINAL = False

        self.CLEAR_CONSOLE_INDEX = self.FPS * 4 # every 4 seconds
        self.FPS_MOD = 1
        self.PIXEL_MOD = 25

        self.run(self.cap)

    def run(self, cap):
        INDEX = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if INDEX % self.FPS_MOD != 0:
                continue
            try:
                img = PIL.Image.fromarray(frame)
                img.resize((self.frame_width, self.frame_height))
            except:
                break
            AF = AsciiFormatter(img, self.ASCII_RESOLUTION_PERCENTAGE, self.PIXEL_MOD, self.BY_CHAR_OR_LINE, self.FONT_SCALE)
            
            if INDEX % self.CLEAR_CONSOLE_INDEX == 0:
                AF.cls() # clear console

            ascii_image = AF.image_to_ascii(self.frame_width, self.frame_height)
            image = AF.ascii_to_image(ascii_image, self.frame_width, self.frame_height)
            self.out.write(image) # write frame to cv2 VideoWriter

            if self.PRINT_IN_CONSOLE:
                print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
                print(ascii_image)
            if self.SHOW_ORIGINAL:
                original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                original_frame = cv2.resize(original_frame, (self.frame_width, self.frame_height))
                image = np.concatenate((original_frame, image), axis=1)
            cv2.imshow(self.APP_NAME, image)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # q -> close window
                break
            INDEX += 1

        cap.release()
        self.out.release()
        cv2.destroyAllWindows()
