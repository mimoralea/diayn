import os
from dm_control.utils import io as resources

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
