Getting Started
###############

Here you will find instructions for installing and setting up EAK-Copilot.

Prerequisites
=============

Before starting, ensure you have the following software installed on your computer:

* **Git:** Needed for source code management. Install from `here <https://git-scm.com/downloads>`_.
* **Docker:** Required for running containers. Install from `here <https://git-scm.com/downloads>`_.
* **Docker Compose:** Necessary for managing multi-container Docker applications. Install from `here <https://git-scm.com/downloads>`_.

Linux users may need to prepend ``sudo`` to Docker commands depending on their Docker configuration.

Installation
============

#. Clone the Repository
    Begin by cloning the EAK-Copilot repository to your local machine to get the necessary project files.

    .. code-block:: console

        git clone https://github.com/CdC-SI/eak-copilot.git
        cd eak-copilot

#. Setting Up Environment Variables
    To use EAK-Copilot, you need to set up some environment variables. Copy the ``.env.example`` file to a new file named ``.env`` and fill in the appropriate values:

    .. code-block:: console

        cp .env.example .env

#. Build Docker Images
    Build the Docker images using the Docker Compose configuration. This step compiles and launches your Docker environment.

    .. code-block:: console

        docker-compose up --build -d

#. Verifying the Installation
    Check the status of the containers to confirm everything is running as expected:

    .. code-block:: console

        docker-compose ps

After the containers are successfully started, navigate to `<http://localhost:3000>`_ in your web browser to access the EAK-Copilot.