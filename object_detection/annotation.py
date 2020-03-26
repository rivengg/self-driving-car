from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np
import cv2


def _round_up(value, n):
   """Rounds up the given value to the next number divisible by n.

   Args:
      value: int to be rounded up.
      n: the number that should be divisible into value.

   Returns:
      the result of value rounded up to the next multiple of n.
   """
   return n * ((value + (n - 1)) // n)


def _round_buffer_dims(dims):
   """Appropriately rounds the given dimensions for image overlaying.

   As per the PiCamera.add_overlay documentation, the source data must have a
   width rounded up to the nearest multiple of 32, and the height rounded up to
   the nearest multiple of 16. This does that for the given image dimensions.

   Args:
      dims: image dimensions.

   Returns:
      the rounded-up dimensions in a tuple.
   """
   width, height = dims
   return _round_up(width, 32), _round_up(height, 16)


class Annotator:
   def __init__(self, img_size, default_color=None):
      self._dims = img_size
      self._buffer_dims = _round_buffer_dims(self._dims)
      self._buffer = Image.new('RGB', self._buffer_dims)
      self._overlay = None
      self._draw = ImageDraw.Draw(self._buffer)
      self._default_color = default_color or (0xFF, 0, 0, 0xFF)

   def update(self, img):
      anno = np.asarray(self._buffer)
      out_img = cv2.addWeighted(img,0.5,anno,0.5,0)
      cv2.imshow('frame', out_img)


   def clear(self):
      """Clears the contents of the overlay, leaving only the plain background."""
      self._draw.rectangle((0, 0) + self._dims, fill=(0, 0, 0, 0x00))

   def bounding_box(self, rect, outline=None, fill=None):
      """Draws a bounding box around the specified rectangle.

      Args:
         rect: (x1, y1, x2, y2) rectangle to be drawn, where (x1, y1) and (x2, y2)
            are opposite corners of the desired rectangle.
         outline: PIL.ImageColor with which to draw the outline (defaults to the
            Annotator default_color).
         fill: PIL.ImageColor with which to fill the rectangle (defaults to None,
            which will *not* cover up drawings under the region).
      """
      outline = outline or self._default_color
      self._draw.rectangle(rect, fill=fill, outline=outline)

   def text(self, location, text, color=None):
      """Draws the given text at the given location.

      Args:
         location: (x, y) point at which to draw the text (upper left corner).
         text: string to be drawn.
         color: PIL.ImageColor to draw the string in (defaults to the Annotator
            default_color).
      """
      color = color or self._default_color
      font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 22)
      self._draw.text(location, text,font=font, fill=color)

