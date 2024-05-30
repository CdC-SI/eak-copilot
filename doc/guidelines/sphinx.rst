Sphinx documentation
####################

Our documentation website is generated with the Sphinx framework and implements the Furo theme.

Here, you will find some documentation to get you started when contributing to this website.
For further information, please refer to `Sphinx <https://www.sphinx-doc.org/en/master/index.html>`_ and `Furo <https://pradyunsg.me/furo/>`_ documentation.

Local build
===========

Prerequisites
    * Python 3.9+

#. Clone the repository
    Begin by cloning the EAK-Copilot repository to your local machine to get the necessary project files.

    .. code-block:: console

        git clone https://github.com/CdC-SI/eak-copilot.git
        cd eak-copilot

#. Install dependencies
    ``doc/requirement.txt`` contains Sphinx and several Sphinx extension packages that we use within the project.
    The other files contain packages required for building the Copilot project, and thus for building the Sphinx documentation.

    .. code-block:: console

        pip install -r doc/requirements.txt
        pip install -r autocomplete/requirements.txt
        pip install -r indexing/requirements.txt
        pip install -r rag/requirements.txt

#. Run the build
    .. code-block:: console

        sphinx-build -M html doc _build

An ``index.html`` document can now be found in the ``_build/html/`` directory. You can open it with your favorite browser and start exploring!


Structure
=======================

Files related to the documentation can be found inside the ``doc/`` directory.

Currently, our /doc folder is structured as follows.

.. code:: none

    doc
    ├── _static
    │   ├ css
    │   └ img
    ├── guidelines
    │   ├ sphinx.rst <--- you are here
    │   └ ...
    ├── introduction
    │   └ ...
    ├── documentation
    │   └ ...
    ├── conf.py
    ├── index.rst
    └── ...

| **conf.py** is the Sphinx configuration files.
| **_static/** contains custom static files, such as stylesheets or images.

All the remaining folders contains ``.rst`` or ``.md`` files only, each file represents a webpage.

| **index.rst** is the website homepage.
| **intro/** contains pages that introduce the Copilot project.
| **guidelines/** contains tutorials and best practices guidelines.
| **topics/** contains Copilot open sourced code documentation.

Especially, **index.rst** contains several "Toc trees" which define the architecture of the website and build the sidebar on the left.
Each ``toctree`` defines a theme. Currently, the website is split into 3 of theme: Introduction, Guidelines, and Documentation.

Thus, there are 3 ``toctree`` defined in doc/index.rst, one for each theme, whose each structure can be visualized as below.

.. tab:: Introduction

    .. code:: none

        Introduction
        ├── introduction/intro (EAK-Copilot)
        ├── introduction/start (Getting Started)
        └── introduction/udpate (Updates)

.. tab:: Guidelines

    .. code:: none

        Introduction
        ├── guidelines/opensource (Open-Source collaboration)
        ├── guidelines/documentation (Documentation standards)
        └── guidelines/sphinx (Sphinx documentation) <--- you are here

.. tab:: Documentation

    .. code:: none

        Documentation
        ├── documentation/api (API)
        ├── documentation/indexing/index (IndexingPipeline)
        │   ├ crawler
        │   ├ scraper
        │   ├ parser
        │   └ chunker
        ├── documentation/survey/index (SurveyPipeline)
        │   └ api (Survey pipeline API)
        ├── documentation/rag/index (RAG)
        │   └ retrieval (RetrievalPipeline)
        ├── documentation/chatbot/index (Chatbot)
        │   └ generation (GenerationPipeline)
        ├── documentation/autocomplete/index (Autocomplete)
        │   ├ exactmatch
        │   ├ fuzzyserach
        │   └ similaritysearch
        └── documentation/gui/index (GUI)
            └ gui

In particular, we can observe the Documentation ``toctree`` pointing at several index.rst files, each containing its own ``toctree`` pointing at subsections.

We can also observe that ``toctree`` has its corresponding directory, inside which we can find its .rst files.


reStructuredText (reST)
=======================

Sphinx uses by default the reStructuredText (reST) markup language, which filename extension is ``.rst``.
You can find the full documentation `here <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_, we provide below a short introduction to this language.

We also recommend having a look at the `Furo's version of the markup documentation <https://pradyunsg.me/furo/reference/>`_, which is the Sphinx theme we are using.
Especially, the `admonitions <https://pradyunsg.me/furo/reference/admonitions/>`_ can be useful and enhance your code descriptions.

Headers convention
------------------

The underline and optional overline with a punctuation character needs to be at least as long as the text.
Normally, there are no heading levels assigned to specific character, but the recommended way is as follow.

.. code:: RST

    ####################################
    Part -- Number Signs above and below
    ####################################

    ************************************
    Chapter -- Asterisks above and below
    ************************************

    Title -- Number Signs
    #####################

    Suptitle -- Asterisks
    *********************

    Section -- Equal Signs
    ======================

    Subsection -- Hyphens
    ---------------------

    Subsubsection -- Circumflex
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Paragraph -- Double Quotes
    """"""""""""""""""""""""""

Inline markup
-------------

.. code:: RST

    *italic text*
    **bold text**
    ``code samples``

Lists
-----

.. code:: RST

    * This is a bulleted list.
    * It has two items, the second
      item uses two lines,

      * and a nested list
      * and some subitems

    * the parent list continues

    1. This is a numbered list.
    2. It has two items too.

    #. This is a numbered list.
    #. It has two items too.

Note that both ``*`` and ``-`` signs work for a bulleted list.

Hyperlinks
----------

.. code:: RST

    Hyperlinks can take various forms, so here's a list of them:

    - standalone hyperlink: https://python.org/
    - hyperlink using references: `link <link>`__
    - hyperlink with inline URL: `link <https://python.org/>`_
    - hyperlink to a different page: :doc:`link <../quickstart>`
    - hyperlink to a specific API element: :class:`pathlib.Path`

    .. _link: https://python.org/

Image
-----

.. code:: RST

    .. image:: https://source.unsplash.com/200x200/daily?cute+animals

Code Blocks
-----------

.. code:: RST

    Below is a code blocks::

        Indenting content by 4 spaces, after a line ends with "::".
        This will have default syntax highlighting (highlighting a few words and "strings").

    .. code::

        You can also use the code directive, or an alias: code-block, sourcecode.
        This will have default syntax highlighting (highlighting a few words and "strings").

    .. code:: python

        print("And with the directive syntax, you can have syntax highlighting.")

    .. code:: none

        print("Or disable all syntax highlighting.")

API documentation
-----------------

.. code:: RST

    .. automodule:: indexing.app.main
       :members:

    .. autoclass:: web_scraper.WebScraper
        :members:

Markdown (MyST)
===============

Although reStructuredText is the base language for Sphinx, through the `MyST extension <https://myst-parser.readthedocs.io/en/latest/intro.html>`_, it also supports ``.md`` files.
You can use ``.md`` if you feel more comfortable with it.

Read their documentation for Sphinx related functionalities, and also `Furo's documentation <https://pradyunsg.me/furo/reference/>`_ which specifies MyST usages in the second tab of their code blocks.