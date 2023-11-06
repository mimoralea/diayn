import os
from dm_control.utils import io as resources

from lxml import etree

_ROOT = os.path.dirname(__file__)
_SCENE_DIR = os.path.join(_ROOT, "a1_scene")
_MODEL_DIR = os.path.join(_ROOT, "a1_model")
_MODEL_ASSET_DIR = os.path.join(_ROOT, 'a1_assets')
_FILENAMES = [
    "materials.xml",
    "skybox.xml",
    "visual.xml",
]


_SCENE_ASSETS = {
  filename: resources.GetResource(os.path.join(_SCENE_DIR, filename))
          for filename in _FILENAMES
}

_ASSETS = _SCENE_ASSETS.copy()
_, _, filenames = next(resources.WalkResources(_MODEL_ASSET_DIR))
for filename in filenames:
    _ASSETS[filename] = resources.GetResource(os.path.join(_MODEL_ASSET_DIR, filename))

def get_model_and_assets(model_filename="a1.xml"):
  """Reads a model XML file and returns its contents as a string."""
  
  xml_string = resources.GetResource(os.path.join(_MODEL_DIR, model_filename))
  parser = etree.XMLParser(remove_blank_text=True)
  mjcf = etree.XML(xml_string, parser)
  
  return etree.tostring(mjcf, pretty_print=True), _ASSETS
