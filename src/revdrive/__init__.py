"""revdrive: a one-pass decision model drives a car through a cone course from what a camera sees.

The model (rev, reading a served vision model) only answers where the course is;
a controller drives there. `revdrive.cli` is the command line, `revdrive.runner`
runs a course, `revdrive.sim` is the world, `revdrive.render` draws the cameras,
`revdrive.point` turns answers into driving, `revdrive.web` is the viewer.
"""

__version__ = "0.1.0"
