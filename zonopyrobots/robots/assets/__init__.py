import os as _os
from urchin import URDF as _URDF
_basedirname = _os.path.dirname(__file__)

class files:
    '''Paths to URDF files for the robots included. All paths are absolute.'''
    def __init__(self):
        '''This class is not meant to be instantiated.'''
        raise Exception("This class is not meant to be instantiated.")

    KinovaGen3 = _os.path.join(_basedirname, 'robots/kinova_arm/gen3.urdf')
    '''URDF file for the Kinova Gen3 robot arm.'''

class urdfs:
    '''URDF objects for the files included. All URDF objects are loaded lazily and share the same name as in :class:`files`.
    If any files are added to the files object, call refresh_assets() to update them in the urdfs object.'''
    def __init__(self):
        '''This class is not meant to be instantiated.'''
        raise Exception("This class is not meant to be instantiated.")

    @staticmethod
    def _get_urdf_property_gen(key, file, doc=None):
        '''Generator for a property that lazily loads a URDF object from a file.'''
        def get_urdf_internal(cls):
            ret = getattr(cls, f"_{key}_internal", None)
            if ret is None:
                ret = _URDF.load(file)
                setattr(cls, f"_{key}_internal", ret)
            return ret
        
        if doc is not None:
            get_urdf_internal.__doc__ = doc
        return property(get_urdf_internal)
    
    @classmethod
    def register_urdf(cls, name, file, doc=None):
        '''Registers a URDF object for a file.'''
        setattr(cls, name, classmethod(cls._get_urdf_property_gen(name, file, doc=doc)))
    
    @classmethod
    def refresh_assets(cls):
        '''Refreshes the URDF objects for the files in the files classobject.'''
        for key in files.__dict__:
            if not key.startswith('__') and key not in files.__dict__:
                docstring = getattr(files, key).__doc__
                file = getattr(files, key)
                cls.register_urdf(
                    key,
                    file,
                    doc=f"Loaded urchin URDF object from the {docstring}"
                )

urdfs.refresh_assets()
'''Initializes the URDF objects for the files included. This is called automatically when the module is imported.'''

# __all__ = ["assets", "urdfs"]
