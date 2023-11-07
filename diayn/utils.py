import os
from dm_control.utils import io as resources
import cv2
import uuid

from lxml import etree

_UNITREE_A1 = os.path.join(os.path.dirname(__file__), "unitree_a1")
_ASSET_DIR = os.path.join(_UNITREE_A1, "assets")
_MODEL_DIR = os.path.join(_UNITREE_A1, "model")
_SCENE_DIR = os.path.join(_UNITREE_A1, "scene")
_FILENAMES = [
    "materials.xml",
    "skybox.xml",
    "visual.xml",
]


_ALL_ASSETS = {
  filename: resources.GetResource(os.path.join(_SCENE_DIR, filename))
          for filename in _FILENAMES
}

_, _, filenames = next(resources.WalkResources(_ASSET_DIR))
for filename in filenames:
    _ALL_ASSETS[filename] = resources.GetResource(os.path.join(_ASSET_DIR, filename))

def get_model_and_assets(model_filename="a1.xml"):
  """Reads a model XML file and returns its contents as a string."""
  
  xml_string = resources.GetResource(os.path.join(_MODEL_DIR, model_filename))
  parser = etree.XMLParser(remove_blank_text=True)
  mjcf = etree.XML(xml_string, parser)
  return etree.tostring(mjcf, pretty_print=True), _ALL_ASSETS


class OpenCVImageViewer():
    """A simple OpenCV highgui based dm_control image viewer

    This class is meant to be a drop-in replacement for
    `gym.envs.classic_control.rendering.SimpleImageViewer`
    """

    def __init__(self, *, escape_to_exit=False):
        """Construct the viewing window"""
        self._escape_to_exit = escape_to_exit
        self._window_name = str(uuid.uuid4())
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        self._isopen = True

    def __del__(self):
        """Close the window"""
        cv2.destroyWindow(self._window_name)
        self._isopen = False

    def imshow(self, img):
        """Show an image"""
        # Convert image to BGR format
        cv2.imshow(self._window_name, img[:, :, [2, 1, 0]])
        # Listen for escape key, then exit if pressed
        if cv2.waitKey(1) in [27] and self._escape_to_exit:
            exit()

    @property
    def isopen(self):
        """Is the window open?"""
        return self._isopen

    def close(self):
        pass