# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import shutil

from sphinx_gallery.scrapers import figure_rst
from sphinx_gallery.sorting import FileNameSortKey
import sphinx_rtd_theme

# -*- coding: utf-8 -*-
#
# NumPyro Tutorials documentation build configuration file, created by
# sphinx-quickstart on Tue Oct 31 11:33:17 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.intersphinx',
              'sphinx.ext.todo',
              'sphinx.ext.mathjax',
              'sphinx.ext.githubpages',
              'nbsphinx',
              'sphinx.ext.autodoc',
              'IPython.sphinxext.ipython_console_highlighting',
              'sphinx_gallery.gen_gallery',
              ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ['.rst', '.ipynb']

# do not execute cells
nbsphinx_execute = 'never'

# allow errors because not all tutorials build
nbsphinx_allow_errors = True

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'NumPyro Tutorials'
copyright = u'2019, Uber Technologies, Inc'
author = u'Uber AI Labs'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

version = ''

if 'READTHEDOCS' not in os.environ:
    # if developing locally, use numpyro.__version__ as version
    from numpyro import __version__  # noqaE402
    version = __version__

# release version
release = version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['.ipynb_checkpoints', 'logistic_regression.ipynb',
                    'examples/*ipynb', 'examples/*py']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# extend timeout
nbsphinx_timeout = 120

# -- Options for gallery --------------------------------------------------

# examples with order
EXAMPLES = [
   'baseball.py',
   'bnn.py',
   'funnel.py',
   'gp.py',
   'ucbadmit.py',
   'hmm.py',
   'neutra.py',
   'ode.py',
   'sparse_regression.py',
   'stochastic_volatility.py',
   'vae.py',
]


class GalleryFileNameSortKey(FileNameSortKey):
    def __call__(self, filename):
        if filename in EXAMPLES:
            return "{:02d}".format(EXAMPLES.index(filename))
        else:  # not in examples list, sort by name
            return "99" + filename


# Adapted from https://sphinx-gallery.github.io/stable/advanced.html#example-2-detecting-image-files-on-disk
#
# Custom images can be put in _static/img folder, with the pattern
#   sphx_glr_[name_of_example]_1.png
# Note that this also displays the image in the example page.
# To not display the image, we can add the following lines
# at the end of __call__ method:
#   if "sparse_regression" in images_rst:
#       images_rst = ""
#   return images_rst
#
# If there are several images for an example, we can select
# which one to be the thumbnail image by adding a comment
# in the example script
#   # sphinx_gallery_thumbnail_number = 2
class PNGScraper(object):
    def __init__(self):
        self.seen = set()

    def __repr__(self):
        return 'PNGScraper'

    def __call__(self, block, block_vars, gallery_conf):
        # Find all PNG files in the directory of this example.
        pngs = sorted(glob.glob(os.path.join(os.path.dirname(__file__), '_static/img/sphx_glr_*.png')))

        # Iterate through PNGs, copy them to the sphinx-gallery output directory
        image_names = list()
        image_path_iterator = block_vars['image_path_iterator']
        for png in pngs:
            if png not in self.seen:
                self.seen |= set(png)
                this_image_path = image_path_iterator.next()
                image_names.append(this_image_path)
                shutil.copy(png, this_image_path)
        # Use the `figure_rst` helper function to generate rST for image files
        images_rst = figure_rst(image_names, gallery_conf['src_dir'])
        return images_rst


sphinx_gallery_conf = {
    'examples_dirs': ['../../examples'],
    'gallery_dirs': ['examples'],
    # slow examples can be added to here to avoid execution
    'filename_pattern': r'(?!hmm_enum)\b\w+.py\b',
    'ignore_pattern': '(minipyro|covtype|__init__)',
    'within_subsection_order': GalleryFileNameSortKey,
    'image_scrapers': ('matplotlib', PNGScraper()),
    'default_thumb_file': 'source/_static/img/pyro_logo_wide.png',
}


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
# logo
html_logo = '_static/img/pyro_logo_wide.png'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    # 'logo_only': True
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_style = 'css/pyro.css'

# html_favicon = '../img/favicon/favicon.ico'


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'NumPyroTutorialsDoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'NumPyroTutorials.tex', u'Numpyro Examples and Tutorials',
     u'Uber AI Labs', 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'NumPyroTutorials', u'Numpyro Examples and Tutorials',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'NumPyroTutorials', u'NumPyro Examples and Tutorials',
     author, 'NumPyroTutorials', 'One line description of project.',
     'Miscellaneous'),
]
