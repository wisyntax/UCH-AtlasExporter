from krita import Krita

from .UCHAtlasExporter import UCHAtlasExporterExtension

app = Krita.instance()
app.addExtension(UCHAtlasExporterExtension(app))
